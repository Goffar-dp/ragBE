import json
import os
import yaml
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
import time
import sys # Added for sys.exit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Configuration ---
LOG_FILE_PATH = "ragbi_user_interactions.jsonl"
GEMINI_API_CONFIG_PATH = "gemini_api.yml" # Your Gemini API key file
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
JUDGE_LLM_MODEL = "gemini-1.5-flash" # Use a powerful model for judging
JUDGE_LLM_TEMPERATURE = 0.1 # Keep temperature low for deterministic judging
GEMINI_RATE_LIMIT_SECONDS = 5 # Adjust based on your Gemini quota if needed

# --- API Key Loading (replicated from ragbi.py for standalone execution) ---
def load_gemini_api_key_for_eval():
    """Load Gemini API key from YAML configuration file for evaluation script."""
    try:
        with open(GEMINI_API_CONFIG_PATH, 'r') as file:
            api_creds = yaml.safe_load(file)
            os.environ['GOOGLE_API_KEY'] = api_creds['api']['api_key']
        print("Gemini API key loaded successfully for evaluation.")
    except Exception as e:
        print(f"Error loading Gemini API key from {GEMINI_API_CONFIG_PATH}: {str(e)}")
        print("Please ensure 'gemini_api.yml' exists and contains your API key.")
        raise

# Load API key at the start of the script
try:
    load_gemini_api_key_for_eval()
except Exception:
    sys.exit("Exiting evaluation script due to missing or invalid Gemini API key.")

# --- Initialize Models ---
print("Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print("Embedding model initialized.")

print("Initializing judge LLM (Gemini)...")
judge_llm = ChatGoogleGenerativeAI(
    model=JUDGE_LLM_MODEL,
    temperature=JUDGE_LLM_TEMPERATURE,
    convert_system_message_to_human=True # Required for older models/system messages
)
print("Judge LLM initialized.")

last_gemini_call_time = time.time()

# --- Evaluation Functions ---

def calculate_relevance(query: str, retrieved_sources: list) -> float:
    """
    Calculates the average cosine similarity between the query embedding
    and the embeddings of the retrieved document content.
    Returns a score between 0 and 1.
    """
    if not retrieved_sources:
        return 0.0 # No sources, so no relevance

    try:
        query_embedding = embedding_model.embed_query(query)
        source_embeddings = []
        for source in retrieved_sources:
            content = source.get("Content Excerpt", "")
            if content:
                source_embeddings.append(embedding_model.embed_query(content))

        if not source_embeddings:
            return 0.0 # No valid content in sources

        # Convert to numpy arrays for cosine similarity calculation
        query_embedding_np = np.array(query_embedding).reshape(1, -1)
        source_embeddings_np = np.array(source_embeddings)

        # Calculate cosine similarity for each source
        similarities = cosine_similarity(query_embedding_np, source_embeddings_np)[0]

        # Return the average similarity as the relevance score
        return float(np.mean(similarities))
    except Exception as e:
        print(f"Error calculating relevance for query '{query}': {e}")
        return 0.0

def calculate_groundedness(query: str, retrieved_sources: list, generated_answer: str) -> str:
    """
    Evaluates groundedness using an LLM-as-a-Judge approach.
    Returns "GROUNDED", "PARTIAL", or "NOT_GROUNDED".
    """
    global last_gemini_call_time

    if not retrieved_sources:
        # If no sources were retrieved, the answer cannot be grounded
        return "NOT_GROUNDED"

    # Combine all retrieved content into a single context string
    full_retrieved_context = "\n\n".join([
        source.get("Content Excerpt", "") for source in retrieved_sources if source.get("Content Excerpt")
    ])

    if not full_retrieved_context.strip():
        # If context is empty after combining, cannot be grounded
        return "NOT_GROUNDED"

    # --- REFINED JUDGE PROMPT ---
    judge_prompt = f"""
You are an expert fact-checker. Your task is to determine if a 'Generated Answer' is fully supported by the 'Retrieved Context'.
You must ONLY use information provided in the 'Retrieved Context' to make your judgment. Do NOT use any outside knowledge.

User Query: {query}

Retrieved Context:
---
{full_retrieved_context}
---

Generated Answer:
---
{generated_answer}
---

Based SOLELY on the 'Retrieved Context', classify the 'Generated Answer' as one of the following:
- 'GROUNDED': If ALL factual claims in the 'Generated Answer' are directly and explicitly verifiable from the 'Retrieved Context'.
- 'PARTIAL': If SOME factual claims in the 'Generated Answer' are verifiable, but others are not, or if there are minor inaccuracies or omissions.
- 'NOT_GROUNDED': If the 'Generated Answer' contains significant factual claims not supported by the 'Retrieved Context', or if it directly contradicts the context.

Your response MUST be a single word: GROUNDED, PARTIAL, or NOT_GROUNDED.
"""

    # Enforce rate limiting for Gemini API calls
    time_since_last_call = time.time() - last_gemini_call_time
    if time_since_last_call < GEMINI_RATE_LIMIT_SECONDS:
        time_to_wait = GEMINI_RATE_LIMIT_SECONDS - time_since_last_call
        # print(f"Rate limiting: Waiting for {time_to_wait:.2f} seconds...") # For debugging
        time.sleep(time_to_wait)

    try:
        # Invoke the judge LLM
        response = judge_llm.invoke(judge_prompt)
        last_gemini_call_time = time.time() # Update last call time

        # Parse the judge's response
        verdict = response.content.strip().upper()
        if verdict in ["GROUNDED", "PARTIAL", "NOT_GROUNDED"]:
            return verdict
        else:
            # Fallback if the LLM doesn't follow instructions perfectly
            print(f"Warning: Judge LLM returned unexpected verdict: '{verdict}' for query: '{query}'. Defaulting to NOT_GROUNDED.") # For debugging
            return "NOT_GROUNDED"
    except Exception as e:
        print(f"Error calling judge LLM for groundedness evaluation for query: '{query}': {e}")
        # If LLM call fails, assume not grounded or handle as an error
        return "ERROR_LLM_JUDGE_FAILED"

# --- Main Evaluation Logic ---
if __name__ == "__main__":
    if not os.path.exists(LOG_FILE_PATH):
        print(f"Error: Log file '{LOG_FILE_PATH}' not found. Please run your Streamlit app to generate interactions.")
        sys.exit(1)

    interactions = []
    try:
        with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                interactions.append(json.loads(line))
        print(f"Loaded {len(interactions)} interactions from '{LOG_FILE_PATH}'.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{LOG_FILE_PATH}': {e}. Please ensure it's valid JSON Lines format.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{LOG_FILE_PATH}': {e}")
        sys.exit(1)

    if not interactions:
        print("No interactions found in the log file. Please interact with your Streamlit app.")
        sys.exit(0)

    relevance_scores = []
    groundedness_results = {"GROUNDED": 0, "PARTIAL": 0, "NOT_GROUNDED": 0, "ERROR_LLM_JUDGE_FAILED": 0}

    print("\nStarting evaluation...")
    for i, interaction in enumerate(interactions):
        query = interaction.get("user_query", "")
        answer = interaction.get("ragbi_answer", "")
        sources = interaction.get("retrieved_sources", [])

        if not query or not answer:
            print(f"Skipping interaction {i+1} due to missing query or answer.")
            continue

        # Calculate Relevance
        relevance = calculate_relevance(query, sources)
        relevance_scores.append(relevance)

        # Calculate Groundedness
        groundedness = calculate_groundedness(query, sources, answer)
        groundedness_results[groundedness] += 1

        # Optional: Print progress
        if (i + 1) % 1 == 0: # Print every interaction for detailed debugging
             print(f"Processed {i + 1}/{len(interactions)}: Query='{query[:50]}...', Relevance={relevance:.4f}, Groundedness='{groundedness}'")


    print("\n--- Evaluation Results ---")

    # Average Relevance
    if relevance_scores:
        avg_relevance = np.mean(relevance_scores)
        print(f"Average Relevance Score (Cosine Similarity): {avg_relevance:.4f} (Higher is better)")
    else:
        print("No relevance scores calculated.")

    # Groundedness Distribution
    total_groundedness_evaluated = sum(groundedness_results.values()) - groundedness_results["ERROR_LLM_JUDGE_FAILED"]
    print("\nGroundedness Distribution:")
    if total_groundedness_evaluated > 0:
        for status, count in groundedness_results.items():
            if status != "ERROR_LLM_JUDGE_FAILED":
                percentage = (count / total_groundedness_evaluated) * 100
                print(f"- {status}: {count} ({percentage:.2f}%)")
        if groundedness_results["ERROR_LLM_JUDGE_FAILED"] > 0:
            print(f"- LLM Judge Failed: {groundedness_results['ERROR_LLM_JUDGE_FAILED']} (These could not be evaluated for groundedness)")
    else:
        print("No groundedness results to display (possibly due to LLM judge errors or no valid interactions).")

    print("\nEvaluation complete. Remember to manually review 'NOT_GROUNDED' cases for deeper insights.")


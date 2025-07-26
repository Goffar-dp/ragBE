import streamlit as st
import pandas as pd
import hashlib
import json # Added for JSON logging
from datetime import datetime # Added for timestamps
import os # Added for file path management

# Assuming ragbi.py is in the same directory
from ragbi import process_documents, get_ragbi_response, GeminiLLM, detect_language

# --- Configuration for Logging ---
# Define the path for  log file.
# It will be created in the same directory as app.py
LOG_FILE_PATH = "ragbi_user_interactions.jsonl"

# --- Logging Function ---
def log_interaction(query: str, answer: str, sources: list, session_id: str = "default_session"):
    """
    Logs a user interaction to a JSON Lines file.
    Each line in the file will be a JSON object representing one interaction.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(), # ISO format for easy parsing
        "session_id": session_id, # Can be expanded for multi-user scenarios
        "user_query": query,
        "ragbi_answer": answer,
        "retrieved_sources": sources
    }
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        # print(f"Logged interaction to {LOG_FILE_PATH}") # For debugging
    except Exception as e:
        print(f"Error logging interaction: {e}")
        
# Set page configuration
st.set_page_config(
    page_title="Bilingual RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "llm" not in st.session_state:
    st.session_state.llm = GeminiLLM()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}

if "file_data_list" not in st.session_state:
    st.session_state.file_data_list = []

# --- Generate a simple session ID for logging ---
# This creates a new session ID each time the app is run or reloaded.
# For more robust user tracking, you'd integrate a proper login system.
if "session_id" not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

# Page title and description
st.title("🌐 Bilingual RAG: Ask Your Documents")
st.markdown("""
Upload documents and ask questions in English or Bengali. The system will retrieve relevant information and generate answers.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload documents to build the knowledge base"
    )

    # Process new files
    if uploaded_files:
        new_files = []
        for file in uploaded_files:
            file_bytes = file.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()

            if file_hash not in st.session_state.processed_files:
                new_files.append((file.name, file_bytes))
                st.session_state.processed_files[file_hash] = file.name
                st.session_state.file_data_list.append((file.name, file_bytes))

        if new_files:
            with st.spinner(f"Processing {len(new_files)} new file(s)..."):
                st.session_state.vectorstore = process_documents(new_files)
            st.success(f"Processed {len(new_files)} file(s)!")

    # Display processed files
    if st.session_state.processed_files:
        st.subheader("Processed Files")
        for file_name in st.session_state.processed_files.values():
            st.markdown(f"✓ {file_name}")

    # Add reset button
    if st.button("Clear All Documents"):
        st.session_state.vectorstore = None
        st.session_state.processed_files = {}
        st.session_state.file_data_list = []
        st.session_state.chat_history = []
        st.success("All documents cleared!")

# Display chat history
st.subheader("Conversation History")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Sources"):
                st.dataframe(pd.DataFrame(msg["sources"]))

# Chat input
query = st.chat_input("Ask a question in English or Bangla...")

if query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Add to history (user query)
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    # Process query
    if not st.session_state.vectorstore:
        error_msg = "Please upload documents first"
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error_msg
        })
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents and generating response..."):
                try:
                    # Get RAG response
                    answer, sources = get_ragbi_response(
                        query,
                        st.session_state.vectorstore,
                        st.session_state.llm
                    )

                    # --- Log the interaction here ---
                    log_interaction(query, answer, sources, st.session_state.session_id)

                    # Display answer
                    st.markdown(answer)

                    # Display sources if available
                    if sources:
                        with st.expander("View Sources"):
                            st.dataframe(pd.DataFrame(sources))

                    # Add to history (assistant response with sources)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"Error processing your query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# language detection info
if st.session_state.chat_history:
    user_queries = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
    if user_queries:
        last_query = user_queries[-1]
        lang = detect_language(last_query)
        lang_name = "Bengali" if lang == "bn" else "English"
        st.sidebar.markdown(f"**Detected Language:** {lang_name}")

# Add system status indicator
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
if st.session_state.vectorstore:
    st.sidebar.success("Vector store ready")
else:
    st.sidebar.warning("No documents processed")

if st.session_state.llm:
    st.sidebar.info(f"Using Gemini: {st.session_state.llm.gemini_model}")

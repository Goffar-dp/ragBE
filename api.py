import os
import sys
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# Add the directory containing ragbi.py to the Python path
# This assumes ragbi.py is in the same directory as api.py
sys.path.append(os.path.dirname(__file__))

# Ensure ragbi.py and gemini_api.yml are in the same directory as api.py
try:
    from ragbi import get_ragbi_response, process_documents, GeminiLLM, load_gemini_api_key
except ImportError as e:
    raise RuntimeError(
        f"Could not import components from ragbi.py. "
        f"Ensure 'ragbi.py' is in the same directory as 'api.py' and all its dependencies are installed. Error: {e}"
    )

# --- Global State for RagBi Components ---
ragbi_vectorstore = None
ragbi_llm = None

# --- FastAPI Application Initialization ---
app = FastAPI(
    title="RagBi API",
    description="A REST API for the RagBi Bilingual RAG System, enabling document upload and conversational queries.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For production, restrict to your frontend's domain.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    global ragbi_llm
    try:
        # Load the Gemini API key using the function from ragbi.py
        load_gemini_api_key()
        # Initialize the Gemini LLM instance
        ragbi_llm = GeminiLLM()
        print("RagBi API: Gemini LLM initialized successfully.")
    except Exception as e:
        # If LLM initialization fails, print an error.
        # The API will still start, but chat functionality will be affected.
        print(f"RagBi API: Failed to initialize Gemini LLM. Chat functionality may be impaired. Error: {e}")

# --- Pydantic Models for Request Bodies ---
# These define the expected structure of incoming JSON data.

class QueryRequest(BaseModel):
    """
    Model for the chat query request.
    """
    query: str

class ChatResponse(BaseModel):
    """
    Model for the chat response, including the answer and source documents.
    """
    answer: str
    sources: List[Dict[str, Any]] # Sources will be a list of dictionaries

# --- Endpoints ---

@app.post(
    "/upload_documents",
    summary="Upload documents to build the RAG knowledge base",
    response_model=Dict[str, str] # Expects a JSON response like {"message": "..."}
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="List of documents (PDF, DOCX, TXT) to upload")
):
    """
    Uploads one or more documents to be processed by the RagBi system.
    The content of these documents will be used to build or update the RAG knowledge base (vector store).
    Supported formats: PDF, DOCX, TXT.
    """
    global ragbi_vectorstore

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded. Please provide at least one document.")

    file_data_list = []
    for file in files:
        try:
            # Read the content of each uploaded file asynchronously
            content = await file.read()
            file_data_list.append((file.filename, content))
        except Exception as e:
            # If there's an error reading a file, return an error response
            raise HTTPException(status_code=500, detail=f"Error reading file '{file.filename}': {e}")

    try:
        # Call the process_documents function from ragbi.py
        # This function handles text extraction, chunking, and building the vector store.
        new_vectorstore = process_documents(file_data_list)
        
        if new_vectorstore:
            # Update the global vector store instance
            ragbi_vectorstore = new_vectorstore
            print(f"RagBi API: Successfully processed {len(file_data_list)} documents and updated knowledge base.")
            return {"message": f"Successfully processed {len(file_data_list)} documents and updated knowledge base."}
        else:
            # If process_documents returns None, it means no documents were processed or an error occurred.
            raise HTTPException(status_code=500, detail="Failed to build vector store. No documents processed or an error occurred during processing.")
    except Exception as e:
        # Catch any other exceptions during document processing
        print(f"RagBi API: Error during document processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {e}")

@app.post(
    "/chat",
    summary="Get a conversational response from the RAG system",
    response_model=ChatResponse # Uses the defined ChatResponse model for output
)
async def chat(request: QueryRequest):
    """
    Sends a user query to the RagBi RAG system.
    The system will retrieve relevant information from the uploaded documents
    and generate a conversational response using the Gemini LLM.
    Returns the generated answer and a list of sources used.
    """
    global ragbi_vectorstore, ragbi_llm

    # Check if the LLM was initialized successfully at startup
    if ragbi_llm is None:
        raise HTTPException(status_code=503, detail="RagBi LLM is not initialized. Please check server logs for errors during startup.")
        
    # Check if documents have been uploaded and the vector store is ready
    if ragbi_vectorstore is None:
        raise HTTPException(status_code=400, detail="Knowledge base not initialized. Please upload documents first via the /upload_documents endpoint.")

    try:
        # Call the get_ragbi_response function from ragbi.py
        # This function performs the RAG logic (retrieval + generation)
        answer, sources = get_ragbi_response(request.query, ragbi_vectorstore, ragbi_llm)
        
        # Return the answer and formatted sources
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        # Catch any exceptions during the RAG response generation
        print(f"RagBi API: Error getting RAG response: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting RAG response: {e}")

# --- Health Check Endpoint ---
@app.get("/health", summary="Health check endpoint")
async def health_check():
    """
    Provides a simple health check for the API.
    Indicates if the API is running and if core components (LLM, Vector Store) are initialized.
    """
    status = {
        "api_status": "healthy",
        "llm_initialized": ragbi_llm is not None,
        "vectorstore_initialized": ragbi_vectorstore is not None,
        "message": "API is running."
    }
    if not ragbi_llm:
        status["message"] = "API is running, but LLM failed to initialize at startup."
    if not ragbi_vectorstore:
        status["message"] = "API is running, but vector store is not initialized. Upload documents to enable RAG."
    return status

# To run this FastAPI app, use the command:
# uvicorn api_main:app --reload
# This will start the server at http://localhost:8000 by default.   
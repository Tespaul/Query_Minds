# filepath: c:\Users\anubh\OneDrive\Desktop\Tes_project\streamlit_interface.py
import streamlit as st
import os
import sys
import numpy as np
from dotenv import load_dotenv
import logging
from data_ingestion import load_documents, chunk_documents
from vector_store import build_vector_store
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from agentic_workflow import route_query, log_routing_decision

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"streamlit_logs.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def initialize_system():
    """Initialize the RAG system with documents and embeddings"""
    with st.spinner('Loading and processing documents...'):
        # Load and chunk documents
        documents = load_documents("data/")
        st.session_state.num_docs = len(documents)
        chunked_data = chunk_documents(documents)
        st.session_state.num_chunks = len(chunked_data)

        # Prepare content and IDs
        contents = [chunk["content"] for chunk in chunked_data]
        ids = [f"doc_{i}" for i in range(len(chunked_data))]

        # Set up embeddings based on selection
        if st.session_state.embedding_type == "mock":
            # Create mock embeddings
            embeddings = [np.random.rand(768).tolist() for _ in range(len(contents))]
            
            # Create a simple mock embedder
            class MockEmbedder:
                def embed_query(self, text):
                    return np.random.rand(768).tolist()
            
            st.session_state.embedder = MockEmbedder()
            logger.info("Using mock embeddings for testing purposes")
        else:
            # Use Gemini embeddings
            try:
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if not gemini_api_key:
                    st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
                    st.stop()
                
                # Create embeddings
                embedder = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=gemini_api_key
                )
                st.session_state.embedder = embedder
                embeddings = embedder.embed_documents(contents)
                logger.info("Using Gemini embeddings")
            except Exception as e:
                st.error(f"Error with Gemini API: {e}")
                st.error("Please check your API key at https://ai.google.dev/")
                st.stop()

        # Build vector store
        build_vector_store(contents, ids, embeddings)
        logger.info("Vector store built successfully")
    
    st.success('System initialized successfully!')

def display_system_info():
    """Display information about the initialized system"""
    st.subheader("System Information")
    st.write(f"- Documents loaded: {st.session_state.num_docs}")
    st.write(f"- Document chunks: {st.session_state.num_chunks}")
    st.write(f"- Embedding type: {st.session_state.embedding_type}")
    
    if st.session_state.history:
        st.subheader("Interaction History")
        for i, (q, a, tool) in enumerate(st.session_state.history):
            with st.expander(f"Q{i+1}: {q[:50]}..."):
                st.write(f"**Query:** {q}")
                st.write(f"**Tool Used:** {tool}")
                st.write(f"**Answer:** {a}")

# Set up the Streamlit interface
def main():
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.title("📚 RAG Assistant with Tools")
    st.write("""
    Ask questions about your documents or use built-in tools:
    - Calculator: Use keywords like 'calculate', 'compute', or include math expressions
    - Dictionary: Use keywords like 'define', 'meaning', or 'what does X mean'
    - Otherwise, your query will be processed through the RAG pipeline
    """)
    
    # Sidebar for configuration and system info
    with st.sidebar:
        st.header("Configuration")
        
        if not st.session_state.initialized:
            st.session_state.embedding_type = st.radio(
                "Select embedding type:",
                ["gemini", "mock"],
                captions=["Google Gemini API (requires API key)", "Random vectors (for testing)"]
            )
            
            if st.button("Initialize System"):
                initialize_system()
                st.session_state.initialized = True
        else:
            display_system_info()
            
            if st.button("Reset System"):
                st.session_state.initialized = False
                st.session_state.history = []
                st.experimental_rerun()
    
    # Main interface for queries
    if st.session_state.initialized:
        query = st.text_input("Enter your query:", placeholder="e.g., What's in our FAQ? or calculate 2+2*3")
        
        if st.button("Submit") and query:
            with st.spinner("Processing query..."):
                # Route the query to the appropriate tool or RAG pipeline
                answer = route_query(query, st.session_state.embedder)
                
                # Determine which tool was used
                if "calculate" in query.lower() or any(c in query for c in "+-*/()0123456789"):
                    tool_used = "Calculator"
                    log_routing_decision(query, "calculator")
                elif any(word in query.lower() for word in ["define", "meaning", "definition"]):
                    tool_used = "Dictionary"
                    log_routing_decision(query, "dictionary")
                else:
                    tool_used = "RAG Pipeline"
                    log_routing_decision(query, "rag")
                
                # Add to history
                st.session_state.history.append((query, answer, tool_used))
            
            # Display result
            st.subheader("Answer")
            st.info(answer)
            st.caption(f"Tool used: {tool_used}")
    else:
        st.info("Please initialize the system using the sidebar options.")

if __name__ == "__main__":
    main()
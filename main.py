from data_ingestion import load_documents, chunk_documents
from vector_store import build_vector_store
from demo_interface import start_cli
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv
import logging
import datetime

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"agent_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()
    logger.info("Environment variables loaded.")
    
    # Load and chunk documents
    logger.info("Loading documents...")
    documents = load_documents("data/")
    logger.info(f"{len(documents)} documents loaded from 'data/'.")
    logger.info("Chunking documents...")
    chunked_data = chunk_documents(documents)
    logger.info(f"Documents chunked into {len(chunked_data)} pieces.")

    # Prepare content and IDs
    contents = [chunk["content"] for chunk in chunked_data]
    ids = [f"doc_{i}" for i in range(len(chunked_data))]
    logger.info(f"Prepared {len(contents)} contents and {len(ids)} IDs for vector store.")
    
    # Generate embeddings
    use_mock = input("Use mock embeddings for testing? (y/n): ").lower() == 'y'
    
    if use_mock:
        logger.info("Using mock embeddings for testing purposes...")
        # Create mock embeddings (768 dimensions to match Gemini's embedding size)
        embeddings = [np.random.rand(768).tolist() for _ in range(len(contents))]
        # Create a simple mock embedder that returns random vectors
        class MockEmbedder:
            def embed_query(self, text):
                return np.random.rand(768).tolist()
        embedder = MockEmbedder()
        logger.info("Mock embedder created and mock embeddings generated.")
    else:
        logger.info("Using Gemini embeddings...")
        try:
            gemini_api_key = os.getenv("GEMINI_API_KEY") 
            
            logger.info(f"Using Gemini API key: {gemini_api_key[:5]}...{gemini_api_key[-5:]}")

            # Create embeddings
            embedder = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=gemini_api_key
            )
            logger.info("Successfully created embedder with Gemini API")
            embeddings = embedder.embed_documents(contents)
            logger.info(f"Generated {len(embeddings)} embeddings.")
        except Exception as e:
            logger.error(f"Error with Gemini API: {e}")
            logger.error("Please check your API key at https://ai.google.dev/")
            sys.exit(1)
    
    # Build vector store
    logger.info("Building vector store...")
    build_vector_store(contents, ids, embeddings)  # collection creation happens inside
    logger.info("Vector store built successfully.")

    # Launch CLI for Q&A
    logger.info("Launching CLI for Q&A...")
    start_cli(embedder)  # pass embedder to CLI for query embeddings

if __name__ == "__main__":
    main()

from chromadb import Client
from chromadb.config import Settings
import numpy as np
import logging
import datetime
import os

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

def retrieve_documents(query, embedder=None, top_k=3):
    """
    Retrieves relevant documents from the vector store.
    
    Args:
        query: The user's query
        embedder: The embedding model to use (if provided)
        top_k: Number of documents to retrieve
    """
    logger.info(f"Retrieving documents for query: '{query}' with top_k={top_k}")
    # Setup ChromaDB client
    settings = Settings(persist_directory="db", anonymized_telemetry=False)
    client = Client(settings)
    collection = client.get_collection("rag_assistant")
    logger.info("ChromaDB client and collection initialized.")

    # Embed the query using the provided embedder or fall back to default
    if embedder:
        try:
            # Use the provided embedder
            logger.info("Using provided embedder for query embedding.")
            query_embedding = embedder.embed_query(query)
            query_embeddings = [query_embedding]
        except Exception as e:
            logger.warning(f"Could not use provided embedder: {e}. Falling back to ChromaDB's default embedding.")
            # Fallback to using ChromaDB's default embedding
            query_embeddings = None
    else:
        logger.info("No embedder provided. Using ChromaDB's default embedding.")
        # Use ChromaDB's default embedding
        query_embeddings = None

    # Query the collection
    if query_embeddings:
        logger.info("Querying collection with provided embeddings.")
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )
    else:
        logger.info("Querying collection with query text, letting ChromaDB handle embedding.")
        # Let ChromaDB handle the embedding
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
    
    retrieved_docs = results["documents"][0] if results and results["documents"] else []
    logger.info(f"Retrieved {len(retrieved_docs)} documents.")
    return retrieved_docs

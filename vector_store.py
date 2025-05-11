from chromadb import Client
from chromadb.config import Settings

def build_vector_store(documents, ids, embeddings=None):
    settings = Settings(
        persist_directory="db",        # You can change this directory
        anonymized_telemetry=False
    )

    client = Client(settings)
    collection = client.get_or_create_collection(name="rag_assistant")

    # Check if we have pre-computed embeddings
    if embeddings:
        collection.add(
            documents=documents,
            ids=ids,
            embeddings=embeddings
        )
    else:
        # Let ChromaDB compute embeddings internally
        collection.add(
            documents=documents,
            ids=ids
        )

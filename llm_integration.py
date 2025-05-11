# LLM API call logic (e.g. OpenAI)
import os
from dotenv import load_dotenv
import openai

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

from chromadb import Client
from chromadb.config import Settings

def call_llm(prompt, api_key):
    """
    Call the LLM (e.g., OpenAI GPT) with a given prompt and return the response.
    """
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Replace with "gpt-4" if needed
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response["choices"][0]["message"]["content"].strip()

class LLMIntegration:
    def __init__(self):
        # Initialize Chroma Client with updated settings
        self.settings = Settings(
            persist_directory="db",  # Directory where the data will be stored
            anonymized_telemetry=False  # Optional: Disable telemetry
        )
        
        # Initialize the Chroma client
        self.client = Client(self.settings)

        # Create or get the collection
        self.collection = self.client.get_or_create_collection("rag_assistant")

    def query_vector_store(self, query_embedding):
        """
        Queries the vector store using the embedding of the query and returns the results.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3  # You can adjust this value for the number of results you want
        )
        return results

    def llm_query(self, query):
        """
        Takes the input query, converts it to an embedding, and queries the vector store.
        Returns the top results.
        """
        # Here you'd integrate your LLM model to convert the query into an embedding
        query_embedding = self.convert_to_embedding(query)
        
        # Query the vector store using the generated embedding
        results = self.query_vector_store(query_embedding)
        
        # Optionally, you could pass the results through your LLM to generate more informative answers.
        return results

    def convert_to_embedding(self, query):
        """
        Convert the input query into an embedding using your LLM.
        This is a placeholder method - replace it with actual LLM embedding generation.
        """
        # For now, this is a dummy embedding generation. Replace this with real LLM embedding.
        # E.g., using OpenAI's model or any other LLM of your choice
        return [0.2] * 768  # Placeholder for an actual embedding from LLM

# Example usage
if __name__ == "__main__":
    llm_integration = LLMIntegration()

    # Example query
    query = "What is AI?"
    
    # Get the top results based on the query
    results = llm_integration.llm_query(query)
    print(results)

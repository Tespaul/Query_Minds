from retrieval import retrieve_documents
from llm import call_llm
from agentic_workflow import route_query, log_routing_decision

def start_cli(embedder=None):
    """
    Starts the command-line interface for interacting with the RAG assistant.
    
    Args:
        embedder: The embedding model to use for embedding queries
    """
    print("Welcome to the RAG Assistant CLI!")
    print("You can ask me questions about our company and products.")
    print("You can also use me as a calculator (e.g., 'calculate 2+2')")
    print("Or ask for definitions (e.g., 'define intelligence')")
    
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        else:
            print(f"Processing query: {query}")
            
            # Route the query to the appropriate tool or RAG pipeline
            answer = route_query(query, embedder)
            
            # Log the routing decision for analysis
            if "calculate" in query.lower() or any(c in query for c in "+-*/()0123456789"):
                log_routing_decision(query, "calculator")
            elif any(word in query.lower() for word in ["define", "meaning", "definition"]):
                log_routing_decision(query, "dictionary")
            else:
                log_routing_decision(query, "rag")
            
            # Print the answer
            print(f"\nAnswer: {answer}\n")

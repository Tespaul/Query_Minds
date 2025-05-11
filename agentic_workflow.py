import re
import logging
import os
from tools.calculator import evaluate_expression
from tools.dictionary import define_word
from retrieval import retrieve_documents
from llm import call_llm

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Get logger
logger = logging.getLogger(__name__)

def route_query(query, embedder=None):
    """
    Routes the query to the appropriate tool or RAG pipeline based on keywords.
    
    Args:
        query: The user's query
        embedder: The embedding model to use for embedding queries in RAG
        
    Returns:
        str: The response from the appropriate tool or RAG pipeline
    """
    # Log the routing decision
    logger.info(f"Routing query: {query}")
    
    # Check if query contains calculator keywords
    calculator_pattern = re.compile(r'\b(calculate|compute|evaluate|solve|what is|[-+*/()0-9])\b', re.IGNORECASE)
    if calculator_pattern.search(query) and any(c in query for c in "+-*/()0123456789"):
        logger.info(f"Routing to calculator tool: {query}")
        # Extract the mathematical expression from the query
        # This is a simple extraction - could be improved with more sophisticated parsing
        expression = re.sub(r'[^0-9+\-*/().%^]', '', query)
        if expression:
            return evaluate_expression(expression)
        else:
            return "I couldn't find a valid mathematical expression in your query."
    
    # Check if query contains dictionary keywords
    define_pattern = re.compile(r'\b(define|meaning|definition|what does|mean)\b', re.IGNORECASE)
    if define_pattern.search(query):
        logger.info(f"Routing to dictionary tool: {query}")
        # Extract the word to define - this is a simple approach
        # Look for patterns like "define [word]" or "what does [word] mean"
        words = query.split()
        word_to_define = None
        
        # Try to find the word after "define", "meaning of", etc.
        for i, word in enumerate(words):
            if re.match(define_pattern, word) and i < len(words) - 1:
                word_to_define = words[i + 1].strip('?,.!:;')
                break
                
        # If we found a word, define it
        if word_to_define:
            return define_word(word_to_define)
        else:
            return "I couldn't determine which word you want me to define. Please specify clearly."
    
    # Default: Use RAG pipeline
    logger.info(f"Using RAG pipeline for query: {query}")
    context_docs = retrieve_documents(query, embedder)
    return call_llm(query, context_docs)

# Log route decisions to a file
def log_routing_decision(query, route):
    """Log routing decisions to a file for analysis."""
    with open(os.path.join(log_dir, "decisions.log"), "a") as f:
        f.write(f"{query} -> {route}\n")

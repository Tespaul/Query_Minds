import os

# Define folders
folders = [
    "data",
    "chunks",
    "logs",
    "tools"
]

# Define placeholder files with initial content
files = {
    "data/faq1.txt": "What is your return policy?",
    "data/faq2.txt": "How do I reset my password?",
    "data/faq3.txt": "Where can I track my order?",
    
    "chunks/.gitkeep": "",  # placeholder to keep folder

    "tools/calculator.py": "# Calculator tool logic will go here\n",
    "tools/dictionary.py": "# Dictionary tool logic will go here\n",
    
    "main.py": "# Main entry point for the assistant\n",
    "data_ingestion.py": "# Logic to load and chunk documents\n",
    "vector_store.py": "# Code for Chroma or FAISS index\n",
    "llm_integration.py": "# LLM API call logic (e.g. OpenAI)\n",
    "agentic_workflow.py": "# Agent decision logic to route queries\n",
    "demo_interface.py": "# CLI or Streamlit app interface\n",
    
    "logs/decisions.log": "",

    "requirements.txt": "openai\nchromadb\nlangchain\ntiktoken\npython-dotenv\n",
    
    "README.md": "# RAG-Powered Multi-Agent Q&A Assistant\n\n## Steps to Run:\n1. Install dependencies using `pip install -r requirements.txt`\n2. Set your OpenAI key in `.env`\n3. Run `main.py` or `demo_interface.py`\n"
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files with placeholder content
for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Project scaffold created successfully!")


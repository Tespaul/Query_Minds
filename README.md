# Query_Minds

This project implements a Retrieval-Augmented Generation (RAG) powered Multi-Agent Q&A system, leveraging the power of *Gemini* (Google’s advanced language model). It utilizes multiple agents to perform data retrieval, reasoning, and response generation. The system also integrates a flexible interface for interaction, either through a CLI or a web-based UI built with Streamlit.

## Prerequisites

- *Python 3.7+*
- *Gemini API Key*: Create a .env file and store your Gemini API Key.

---

## Setup Instructions

1. *Clone the repository*:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

2. Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Install the dependencies:

pip install -r requirements.txt


4. Set up your .env file:

Create a .env file in the root of the project directory and add your Gemini API key as follows:

GEMINI_API_KEY=your_api_key_here


File Descriptions

Core Files

main.py:
The entry point of the project. It initializes the data ingestion, embedding process, and starts the appropriate interface (CLI or web UI).

data_ingestion.py:
Responsible for reading content from various .txt files (like DATA1.txt, DATA2.txt) and preparing it for embedding into a vector store.

vector_store.py:
Handles the creation and management of the vector store (e.g., ChromaDB). It stores the embedded representations of the documents.

retrieval.py:
Implements the document retrieval logic by querying the vector store for the most relevant documents based on the user’s query.

llm_integration.py:
Manages interaction with the Gemini API for generating answers. It uses the embeddings from the retrieval step to generate a response.

agentic_workflow.py:
Coordinates different agents in the system, such as the Retriever, Reasoner, and Responder, to collaboratively provide accurate and context-aware answers.


Interfaces

demo_interface.py:
Provides a command-line interface (CLI) for interacting with the Q&A system.

streamlit_interface.py:
(Optional) Provides a web-based interface for interacting with the Q&A system using Streamlit.


Utilities

llm.py:
Defines the methods and logic for embedding queries and documents, and generating responses using the Gemini API.

logger.py:
A custom logging utility used throughout the project for debugging and tracking the system's processes.

calculator.py:
(Optional) A utility script that can be used for any kind of auxiliary calculations or processing.

dictionary.py:
A utility for handling any mappings or definitions, potentially used for managing terminology or synonyms.


Data Files

DATA1.txt, DATA2.txt, DATA3.txt:
Text files containing the raw data used for the Q&A system. These files will be ingested and processed into vector embeddings.

faq1.txt:
An optional FAQ file, which can also be ingested into the system to provide quick answers based on frequently asked questions.


Other Files

requirements.txt:
Contains a list of all the Python packages required to run the project, including dependencies like langchain, chromadb, and streamlit.

setupfinal.py:
A setup script that may be used for initializing any configurations or utilities for the system.



---

Running the System

To run the system, you can use either the CLI or the web interface.

CLI Interface

1. Make sure the .env file is configured with your Gemini API Key.


2. Run the CLI:

python demo_interface.py


3. You will be prompted to input your query, and the system will return a context-aware response based on the vector retrieval and Gemini LLM.



Streamlit Interface

If you prefer a web-based interface, you can run:

streamlit run streamlit_interface.py

This will launch a web application where you can interact with the system via your browser.

Notes

Make sure to follow the setup instructions and configure the .env file with your Gemini API key for the system to function correctly.

The system leverages vector embeddings and semantic search to retrieve the most relevant documents before generating a response using Gemini, so be sure your input data is formatted correctly.

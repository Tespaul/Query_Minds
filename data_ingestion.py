import os

# Option 1: Use a raw string
data_path = r"C:\Users\ACER\OneDrive\Desktop\Rag powered_project\Tes_project\data"

# Option 2: Use forward slashes
# data_path = "C:/Users/ACER/OneDrive/Desktop/Rag powered_project/Tes_project/data"

def load_documents(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The directory '{data_path}' does not exist.")
    if not os.listdir(data_path):
        raise FileNotFoundError(f"The directory '{data_path}' is empty.")
    
    documents = []
    for filename in os.listdir(data_path):
        file_path = os.path.join(data_path, filename)
        if os.path.isfile(file_path):  # Ensure it's a file
            with open(file_path, 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents

def chunk_documents(documents, chunk_size=200):
    """
    Splits documents into smaller chunks for better retrieval.
    """
    chunked_data = []
    for doc in documents:
        for i in range(0, len(doc), chunk_size):
            chunked_data.append({"content": doc[i:i+chunk_size], "source": "source.txt"})
    return chunked_data

from qdrant_client import QdrantClient, models
import glob
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

qdrant_client = QdrantClient(url="http://localhost:6333")
collection_name = "deepresearch-documents"

def initialize_collection():
    """Initializes the Qdrant collection."""
    # Create a collection with the specified parameters
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # Size of the vector
                distance=models.Distance.COSINE,  # Distance metric
            )
        )

def split_file_to_chunks(file_path, chunk_size=1000, chunk_overlap=100):
    """Splits a file into chunks of specified size."""
    with open(file_path, 'rb') as f:
        #read all file text
        file_text = f.read()
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(file_text)
        return chunks
    
def embed_chunk(chunk):
    """Embeds a chunk of text using Azure OpenAI embedding."""
    response = openai.Embedding.create(
        input=chunk,
        engine="text-embedding-3-large",  # Replace with your deployment name
        api_type="azure",
        api_base="https://tima-openai2.openai.azure.com/",  # Replace with your endpoint
        api_version="2023-05-15",
        api_key="0ca5fb4a37704500a22fff6e68c95bda"  # Replace with your API key or use environment variable
    )
    return response['data'][0]['embedding']

def store_document_in_qdrant(chunks):
    """Stores the document chunks in Qdrant."""
    for chunk in chunks:
        # Embed the chunk
        embedding = embed_chunk(chunk)
        # Store the chunk in Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=chunk,  # Use a unique ID for each chunk
                    vector=embedding,
                    payload={"text": chunk}  # Store the text as payload
                )
            ]
        )

def main():
    initialize_collection()
    subfolder = "docs"
    file_names = glob.glob(os.path.join(subfolder, "**", "*.*"), recursive=True)

    all_chunks = []
    for file_path in file_names:
        # Split the file into chunks
        chunks = split_file_to_chunks(file_path)
        all_chunks.extend(chunks)
        print(f"Read {len(chunks)} chunks from {file_path}")

    print(f"Total chunks: {len(all_chunks)}")
    store_document_in_qdrant(all_chunks)
    print("All documents stored in Qdrant.")

if __name__ == "__main__":
    main()
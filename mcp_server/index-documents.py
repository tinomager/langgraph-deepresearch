from qdrant_client import QdrantClient, models
import glob
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI

# Load environment variables from .env if present
from dotenv import load_dotenv
from model import ChunkModel
load_dotenv()

qdrant_client = QdrantClient(url=os.environ["QDRANT_URL"])
collection_name = os.environ["QDRANT_COLLECTION_NAME"]
azure_client = AzureOpenAI(
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

def initialize_collection():
    """Initializes the Qdrant collection."""
    # Create a collection with the specified parameters
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=3072,  # Size of the vector
                distance=models.Distance.COSINE,  # Distance metric
            )
        )

def split_file_to_chunks(file_path, chunk_size=1000, chunk_overlap=100):
    """Splits a file into chunks of specified size."""
    with open(file_path, 'rb') as f:
        #read all file text
        file_text = f.read().decode("utf-8")
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(file_text)
        filename = os.path.basename(file_path)
        # Map each chunk to a ChunkModel instance with chunk number
        chunks = [
            ChunkModel(content=chunk, filename=filename, chunknumber=i)
            for i, chunk in enumerate(chunks)
        ]
        return chunks
    
def embed_chunk(chunk):
    """Embeds a chunk of text using Azure OpenAI embedding."""

    response = azure_client.embeddings.create(input=chunk.content, 
                                              model=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"])

    return response.data[0].embedding

def store_document_in_qdrant(chunks):
    """Stores the document chunks in Qdrant."""
    id_counter = 0
    for chunk in chunks:
        # Embed the chunk
        embedding = embed_chunk(chunk)
        # Store the chunk in Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=id_counter,  # Use a unique ID for each chunk
                    vector=embedding,
                    payload={
                        "content": chunk.content,
                        "filename": chunk.filename,
                        "chunknumber": chunk.chunknumber
                    }
                )
            ]
        )

        id_counter += 1

def main():
    initialize_collection()
    subfolder = os.environ.get("DOCS_SUBFOLDER", "docs")
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
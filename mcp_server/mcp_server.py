from mcp.server.fastmcp import FastMCP
import httpx
import datetime
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
load_dotenv()
import os
from openai import AzureOpenAI
from model import ChunkModel


# Create an MCP server
mcp = FastMCP("RAG MCP Server", stateless_http=True, json_response=True)
qdrant_url = os.environ["QDRANT_URL"]
qdrant_client = QdrantClient(url=qdrant_url)

collection_name = os.environ["QDRANT_COLLECTION_NAME"]
azrue_client = AzureOpenAI(
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

# Get data from internal documents
@mcp.tool()
def get_rag_data(query: str, num_docs: int = 5) -> str:
    """Get data from document knowledge based on the user query"""
    print(f"Received MCP query at tool get_rag_data: {query}")

    results = search_documents(query, num_docs)
    combined_text = "\n\n".join([point.payload.get("content", "") for point in results.points])
    print(f"Query: {query}\n\nResults:\n\n{combined_text}")

    return combined_text

@mcp.tool()
def get_rag_data_with_context(query: str, num_docs: int = 5) -> str:
    """Get data from document knowledge based on the user query"""
    print(f"Received MCP query at tool get_rag_data: {query}")

    results = search_documents(query, num_docs)
    #results = [
    #        ChunkModel(content=point.payload.get("content", ""), filename=point.payload.get("filename", ""), chunknumber=point.payload.get("chunknumber", ""))
    #        for point in results.points
    #    ]
    #print(f"Query: {query}\n\nResults:\n\n{combined_text}")

    result_json = [
        { 
            "content": point.payload.get("content", ""),
            "filename": point.payload.get("filename", ""),
            "chunknumber": point.payload.get("chunknumber", ""),
            "score": point.score 
        }
        for point in results.points
    ]
    return result_json
    
def search_documents(query: str, num_docs: int = 5):
    embedding_response = azrue_client.embeddings.create(
        input=query,
        model=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]
    )
    embedding = embedding_response.data[0].embedding
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=embedding,
        limit=num_docs,
        with_payload=True
    )
    return results

if __name__ == "__main__":
    # Check if Qdrant is reachable before starting the server
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if collection_name not in collection_names:
            print(f"Error: Collection '{collection_name}' not found in Qdrant. Available collections: {collection_names}. Please create the collection before starting the server.")
            exit(1)
    except Exception as e:
        print(f"Error: Could not connect to Qdrant at {qdrant_url}: {e}")
        exit(1)
    mcp.run(transport="streamable-http")

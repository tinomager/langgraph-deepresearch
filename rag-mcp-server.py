# rag-mcp-server.py
from mcp.server.fastmcp import FastMCP
import httpx
import datetime
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
load_dotenv()
import os
from openai import AzureOpenAI


# Create an MCP server
mcp = FastMCP("RAG MCP Server", "1.0.0")
qdrant_client = QdrantClient(url=os.environ["QDRANT_URL"])
collection_name = os.environ["QDRANT_COLLECTION_NAME"]
azrue_client = AzureOpenAI(
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

@mcp.tool()
def get_time_with_prefix():
    """Get the current date and time."""
    return str(datetime.datetime.now())

# Get data from BWI document
@mcp.tool()
def get_rag_data(query: str) -> str:
    """Get data from document knowledge based on the query"""

    embedding_response = azrue_client.embeddings.create(input=query,
                                               model=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"])
    embedding = embedding_response.data[0].embedding
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=embedding,
        limit=5,
        with_payload=True
    )

    # Extract the text from the results
    combined_text = "\n\n".join([point.payload.get("text", "") for point in results.points])
    print(f"Query: {query}\n\nResults:\n\n{combined_text}")

    return combined_text

@mcp.tool()
async def fetch_weather(city: str) -> str:
    """Fetch current weather for a city"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.text
    
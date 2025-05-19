# rag-mcp-server.py
from mcp.server.fastmcp import FastMCP
import httpx
import datetime

# Create an MCP server
mcp = FastMCP("RAG MCP Server", "1.0.0")


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
    """Get data from BWI document based on the query"""
    return f"Your query: {query}!"

@mcp.tool()
async def fetch_weather(city: str) -> str:
    """Fetch current weather for a city"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.text
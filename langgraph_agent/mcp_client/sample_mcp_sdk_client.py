import asyncio
from mcp_client import MCPClient

url = "http://localhost:8000/mcp/stream"
sample_query = "Wie lege ich Wäsche nach?"

async def main():
    client = MCPClient()
    try:
        await client.connect_to_streamable_http_server(url)
        await client.process_query(sample_query)
    finally:
        await client.cleanup()
    print("Finished MCP testrun.")

if __name__ == "__main__":
    asyncio.run(main())

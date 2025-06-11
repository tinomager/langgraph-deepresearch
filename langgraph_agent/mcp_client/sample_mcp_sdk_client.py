import asyncio
from mcp_client import MCPClient
import json

url = "http://localhost:8000/mcp/stream"
sample_query = "Wie lege ich Wäsche nach?"

async def main():
    client = MCPClient()
    try:
        await client.connect_to_streamable_http_server(url)
        result_simple = await client.process_query(sample_query)
        print(f"Result from simple query: {result_simple}")
        result_with_context = await client.process_query_with_context(sample_query)
        print(f"Result from query with context:")
        for chunk in result_with_context.content:
            chunk_json = json.loads(chunk.text)
            print(f"Chunk content: {chunk_json['content'][:100]}")
            print(f"Filename: {chunk_json['filename']}")
            print(f"Chunk number: {chunk_json['chunknumber']}")
            print(f"Score: {chunk_json['score']}")
    finally:
        await client.cleanup()
    print("Finished MCP testrun.")

if __name__ == "__main__":
    asyncio.run(main())

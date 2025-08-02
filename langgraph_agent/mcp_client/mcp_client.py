import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from typing import Optional
from contextlib import AsyncExitStack

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_streamable_http_server(
        self, server_url: str, headers: Optional[dict] = None
    ):
        """Connect to a Streamable HTTP server."""
        self._stream_context = streamablehttp_client(
            url=server_url,
            headers=headers or {}
        )
        read_stream, write_stream, _ = await self._stream_context.__aenter__()  
        self._session_context = ClientSession(read_stream=read_stream, write_stream=write_stream)
        self.session = await self._session_context.__aenter__()
        print(f"Connected to MCP Streamable HTTP server at {server_url}")

    async def process_query(self, query: str, num_docs: int = 5):
        tool_args = {"query": query, "num_docs": num_docs}
        tool_name = "get_rag_data"
        try:
            result = await self.session.call_tool(tool_name, tool_args)
            if not result.isError:
                result_text = result.content[0].text
                return result_text
            else:
                print(f"Error processing query: {query}\nError: {result.error if hasattr(result, 'error') else result}")
                raise Exception(f"Error processing query: {query}")
        except Exception as e:
            import traceback
            print(f"Exception in process_query: {e}\nTraceback:\n{traceback.format_exc()}")
            raise
        
    async def process_query_with_context(self, query: str, num_docs: int = 5):
        tool_args = {"query": query, "num_docs": num_docs}
        tool_name = "get_rag_data_with_context"  
        result = await self.session.call_tool(
            tool_name, tool_args
        )
        if not result.isError:
            return result
        else:
            raise Exception(f"Error processing query: {query}")

    async def cleanup(self):
        """Cleanup resources."""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._stream_context:  
            await self._stream_context.__aexit__(None, None, None)

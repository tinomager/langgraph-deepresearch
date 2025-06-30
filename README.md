# Deep Research Agent with Langgraph

This repository contains an implementation of a Langgraph agent that makes use of an MCP server to retrieve knowledge from the server. The agent can make use of the implemented RAG-tools of the server.

## MCP Server

The MCP server is implemented as a FastMCP server with a backend implementing basic RAG with Qdrant and Azure OpenAI Embeddings.

See [`README.md`](mcp_server/README.md) of the MCP server for the details on how to run the server.

## Langgraph Agent

See [`README.md`](mcp_server/README.md) of the Langgraph agent for the details on how to run the agent.
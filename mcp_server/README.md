# MCP Server

This directory implements a Minimal Context Protocol (MCP) server for Retrieval-Augmented Generation (RAG) using Qdrant as a vector database and Azure OpenAI for embeddings.

## Features

- **Document Chunking & Embedding:** Splits documents into chunks, embeds them using Azure OpenAI, and stores them in Qdrant.
- **MCP Server:** Exposes tools to retrieve relevant document chunks based on user queries.
- **Docker Support:** Includes a command to run Qdrant locally via Docker.

## Quickstart

### 1. Prerequisites

- Python 3.8+
- Docker (for Qdrant)
- Azure OpenAI API access

### 2. Start Qdrant (Vector DB)

```sh
docker run --name qdrant_mcp_container -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```
See [`qdrant-docker-create-command.txt`](mcp_server/qdrant-docker-create-command.txt:1).

### 3. Configure Environment

Copy [`sample.env`](mcp_server/sample.env:1) to `.env` and fill in your credentials.

### 4. Index Documents

Place your documents in the folder specified by `DOCS_SUBFOLDER` in `.env` (default: `docs/`). Then run:

```sh
python index-documents.py
```
See [`index-documents.py`](mcp_server/index-documents.py:1).

### 5. Run the MCP Server

```sh
python mcp_server.py
```
See [`mcp_server.py`](mcp_server/mcp_server.py:1).

## API Tools

- **get_rag_data(query: str, num_docs: int = 5) → str**
  Returns concatenated text of the top-matching document chunks for a query.

- **get_rag_data_with_context(query: str, num_docs: int = 5) → List[dict]**
  Returns a list of dicts with `content`, `filename`, `chunknumber`, and `score` for each chunk.

## Environment Variables

See [`sample.env`](mcp_server/sample.env:1) for all required variables:

- `QDRANT_URL`
- `QDRANT_COLLECTION_NAME`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_EMBEDDING_MODEL`
- `DOCS_SUBFOLDER`
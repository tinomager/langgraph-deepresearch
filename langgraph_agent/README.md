# LangGraph Agent

This module provides a configurable research agent that leverages Large Language Models (LLMs) and an MCP server connection to perform automated research and summarization tasks.

## Main Components

- **agent.py**: Core logic of the research agent. Contains the classes `AgentConfig` (configuration via environment variables) and `ResearchAgent` (controls the research and summarization process). Integrates LLM (Azure OpenAI) and MCP client.
- **helper.py**: Helper functions and configuration classes for the agent logic and the StateGraph.
- **prompts.py**: Contains prompt templates for LLM interaction (e.g., query generation, summarization, reflection).
- **states.py**: Defines the state objects for the workflow (e.g., `SummaryState`, input/output models).
- **mcp_client/**: Implements the MCP client for communication with the MCP server (e.g., `mcp_client.py`, sample client `sample_mcp_sdk_client.py`).
- **sample.env**: Example for required environment variables.

## Typical Workflow

1. **Configuration**: API keys, endpoints, and other parameters are set via environment variables (see `sample.env`).
2. **Agent Initialization**: Instantiate `AgentConfig` and `ResearchAgent`.
3. **Graph-based Workflow**: Using LangGraph, a StateGraph is built that models the steps query generation, research, summarization, reflection, and finalization.
4. **MCP Integration**: The agent uses the MCP client to send search queries to an MCP server and process the results.
5. **Summarization**: The collected sources are summarized using the LLM and can optionally include source references.

## Usage

See the example at the bottom of `agent.py` for a typical invocation:

```python
if __name__ == "__main__":
    from states import SummaryStateInput
    config = AgentConfig()
    agent = ResearchAgent(config)
    graph = agent.build_graph()
    research_input = SummaryStateInput(
        research_topic="Benefits of Miele WTI 360"
    )
    summary = graph.invoke(research_input)
    print(summary["final_summary"])
```

## Requirements

- Python 3.10+
- Dependencies as specified in `pyproject.toml` (e.g., `openai`, `dotenv`, `langgraph`)
- Credentials for Azure OpenAI and MCP server

## License

See LICENSE in the root directory.
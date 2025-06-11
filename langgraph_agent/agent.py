# Data handling
import json
import re  
from typing import Literal
from fastmcp import Client
from mcp_client.mcp_client import MCPClient
import asyncio

from states import SummaryState, SummaryStateInput, SummaryStateOutput
from prompts import (
    query_writer_prompt,
    summarizer_instructions_prompt,
    reflection_instructions_prompt
)
from helper import (
    deduplicate_and_format_sources,
    format_sources
)
from helper import Configuration

# LangChain components
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph

from openai import AzureOpenAI


from dotenv import load_dotenv
load_dotenv()

import os


api_key = os.environ["AZURE_OPENAI_API_KEY"]  
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]  
deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]             
client = AzureOpenAI(
    api_key=api_key,
    api_version=os.environ["AZURE_OPENAI_API_VERSION"], 
    azure_endpoint=endpoint,
)
max_research_loops = int(os.environ["MAX_RESEARCH_LOOPS"])
mcp_server_url = os.environ["MCP_SERVER_URL"]
debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "yes")

def call_llm(messages : list, temperature: float = 0.7, json_response : bool = False):
    """ Call the LLM with the given messages and temperature """
    if json_response:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=temperature,
            response_format= { "type":"json_object" }
        )
    else:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=temperature
        )
    
    return response.choices[0].message.content

# generate query node
def generate_query(state: SummaryState):
    """ Generate a query for web search """

    # Format the prompt
    query_writer_instructions_formatted = query_writer_prompt.format(research_topic=state.research_topic)

    # Generate a query
    result = call_llm(
        [
            {"role": "system", "content": query_writer_instructions_formatted},
            {"role": "user", "content": "Generate a query for research:"}
        ],
        temperature=0,
        json_response=True
    )
    query = json.loads(result)

    print(f"Generated query: {query['query']}/nAspect: {query['aspect']}/nRationale: {query['rationale']}")

    return {"search_query": query['query']}

def mcp_research(state: SummaryState):
    """ Gather information from the web """

    # Search with MCP
    search_results = asyncio.run(do_mcp_research(state))
    return {"sources_gathered": [search_results], "research_loop_count": state.research_loop_count + 1, "mcp_research_results": [search_results]}
    #search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000)
    #return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "mcp_research_results": [search_str]}

async def do_mcp_research(state: SummaryState):
    client = MCPClient()
    result = ""
    try:
        await client.connect_to_streamable_http_server(mcp_server_url)
        result = await client.process_query(state.search_query)
    finally:
        await client.cleanup()

    return result

def summarize_sources(state: SummaryState):
    """ Summarize the gathered sources """

    # Existing summary
    existing_summary = state.final_summary

    # Most recent web research
    most_recent_mcp_research = state.research_results[-1] if state.research_results else ""

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"Extend the existing summary: {existing_summary}\n\n"
            f"Include new search results: {most_recent_mcp_research} "
            f"That addresses the following topic: {state.research_topic}"
        )
    else:
        human_message_content = (
            f"Generate a summary of these search results: {most_recent_mcp_research} "
            f"That addresses the following topic: {state.research_topic}"
        )

    # Run the LLM
    result = call_llm(
        [
            {"role": "system", "content": summarizer_instructions_prompt},
            {"role": "user", "content": human_message_content}
        ],
        temperature=0
    )

    final_summary = result
    return {"final_summary": final_summary}

def reflect_on_summary(state: SummaryState):
    """ Reflect on the summary and generate a follow-up query """

    # Generate a query
    result = call_llm(
        [
            {"role": "system", "content": reflection_instructions_prompt.format(research_topic=state.research_topic)},
            {"role": "user", "content": f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.final_summary}"}
        ],
        temperature=0,
        json_response=True
    )
    follow_up_query = json.loads(result)

    # Overwrite the search query
    return {"search_query": follow_up_query['follow_up_query']}

def finalize_summary(state: SummaryState):
    """ Finalize the summary """

    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.final_summary = f"## Summary\n\n{state.final_summary}\n\n ### Sources:\n{all_sources}"
    return {"final_summary": state.final_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "mcp_research"]:
    """ Route the research based on the follow-up query """

    if state.research_loop_count <= max_research_loops:
        return "mcp_research"
    else:
        return "finalize_summary" 
    
# Add nodes and edges 
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("mcp_research", mcp_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "mcp_research")
builder.add_edge("mcp_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research) # conditional to wether to continue research or finalize
builder.add_edge("finalize_summary", END)

# compile the graph
graph = builder.compile()

# cleaned_summary = re.sub(r'<think>.*?</think>', '', summary['final_summary'], flags=re.DOTALL)
# cleaned_summary = re.sub(r'\n{3,}', '\n\n', cleaned_summary)
# print(cleaned_summary)

# test the agent
research_input = SummaryStateInput(
    research_topic="Benefits of Miele WTI 360"
)
summary = graph.invoke(research_input)

# response format
cleaned_summary = re.sub(r'<think>.*?</think>', '', summary['final_summary'], flags=re.DOTALL)
cleaned_summary = re.sub(r'\n{3,}', '\n\n', cleaned_summary)
print(cleaned_summary)
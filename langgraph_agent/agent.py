
# ...existing code...

class AgentConfig:
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        import os
        self.api_key = os.environ["AZURE_OPENAI_API_KEY"]
        self.endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        self.deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
        self.api_version = os.environ["AZURE_OPENAI_API_VERSION"]
        self.max_research_loops = int(os.environ["MAX_RESEARCH_LOOPS"])
        self.mcp_server_url = os.environ["MCP_SERVER_URL"]
        self.debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "yes")
        # New config: print sources in finalize_summary
        self.print_sources_in_summary = os.environ.get("PRINT_SOURCES_IN_SUMMARY", "False").lower() in ("true", "1", "yes")


class ResearchAgent:
    def __init__(self, config: AgentConfig):
        from openai import AzureOpenAI
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint,
        )
        # Import here to avoid circular imports
        from mcp_client.mcp_client import MCPClient
        self.MCPClient = MCPClient

    def call_llm(self, messages: list, temperature: float = 0.7, json_response: bool = False):
        if json_response:
            response = self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
        else:
            response = self.client.chat.completions.create(
                model=self.config.deployment_name,
                messages=messages,
                temperature=temperature
            )
        return response.choices[0].message.content

    def generate_query(self, state):
        from prompts import query_writer_prompt
        import json
        query_writer_instructions_formatted = query_writer_prompt.format(research_topic=state.research_topic)
        print("\n[generate_query] -- LLM prompt:")
        print(query_writer_instructions_formatted)
        result = self.call_llm(
            [
                {"role": "system", "content": query_writer_instructions_formatted},
                {"role": "user", "content": "Generate a query for research:"}
            ],
            temperature=0,
            json_response=True
        )
        query = json.loads(result)
        print(f"[generate_query] -- Generated query: {query['query']}\nAspect: {query.get('aspect', '')}\nRationale: {query.get('rationale', '')}")
        return {"search_query": query['query']}

    def mcp_research(self, state):
        import asyncio
        print(f"\n[mcp_research] -- Executing MCP research for query: {state.search_query}")
        search_results = asyncio.run(self.do_mcp_research(state))
        print(f"[mcp_research] -- MCP research results: {search_results}")
        return {"sources_gathered": [search_results], "research_loop_count": state.research_loop_count + 1, "mcp_research_results": [search_results]}

    async def do_mcp_research(self, state):
        client = self.MCPClient()
        result = ""
        try:
            await client.connect_to_streamable_http_server(self.config.mcp_server_url)
            result = await client.process_query(state.search_query)
        finally:
            await client.cleanup()
        return result

    def summarize_sources(self, state):
        from prompts import summarizer_instructions_prompt
        existing_summary = state.final_summary
        most_recent_mcp_research = state.research_results[-1] if state.research_results else ""
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
        print("\n[summarize_sources] -- LLM prompt:")
        print(summarizer_instructions_prompt)
        print("[summarize_sources] -- User message:")
        print(human_message_content)
        result = self.call_llm(
            [
                {"role": "system", "content": summarizer_instructions_prompt},
                {"role": "user", "content": human_message_content}
            ],
            temperature=0
        )
        final_summary = result
        print(f"[summarize_sources] -- Final summary: {final_summary}")
        return {"final_summary": final_summary}

    def reflect_on_summary(self, state):
        from prompts import reflection_instructions_prompt
        import json
        prompt = reflection_instructions_prompt.format(research_topic=state.research_topic)
        print("\n[reflect_on_summary] -- LLM prompt:")
        print(prompt)
        print("[reflect_on_summary] -- User message:")
        print(f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.final_summary}")
        result = self.call_llm(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.final_summary}"}
            ],
            temperature=0,
            json_response=True
        )
        follow_up_query = json.loads(result)
        print(f"[reflect_on_summary] -- Follow-up query: {follow_up_query.get('follow_up_query', '')}")
        return {"search_query": follow_up_query['follow_up_query']}

    def finalize_summary(self, state):
        if self.config.print_sources_in_summary:
            all_sources = "\n".join(source for source in state.sources_gathered)
            state.final_summary = f"## Summary\n\n{state.final_summary}\n\n ### Sources:\n{all_sources}"
            print("\n[finalize_summary] -- Final summary with sources:")
        else:
            state.final_summary = f"## Summary\n\n{state.final_summary}"
            print("\n[finalize_summary] -- Final summary (sources omitted):")
        print(state.final_summary)
        return {"final_summary": state.final_summary}

    def route_research(self, state, config):
        if state.research_loop_count <= self.config.max_research_loops:
            return "mcp_research"
        else:
            return "finalize_summary"

    def build_graph(self):
        from states import SummaryState, SummaryStateInput, SummaryStateOutput
        from helper import Configuration
        from langgraph.graph import START, END, StateGraph
        builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
        builder.add_node("generate_query", self.generate_query)
        builder.add_node("mcp_research", self.mcp_research)
        builder.add_node("summarize_sources", self.summarize_sources)
        builder.add_node("reflect_on_summary", self.reflect_on_summary)
        builder.add_node("finalize_summary", self.finalize_summary)
        builder.add_edge(START, "generate_query")
        builder.add_edge("generate_query", "mcp_research")
        builder.add_edge("mcp_research", "summarize_sources")
        builder.add_edge("summarize_sources", "reflect_on_summary")
        builder.add_conditional_edges("reflect_on_summary", self.route_research)
        builder.add_edge("finalize_summary", END)
        return builder.compile()


if __name__ == "__main__":
    import re
    from states import SummaryStateInput
    config = AgentConfig()
    agent = ResearchAgent(config)
    graph = agent.build_graph()
    research_input = SummaryStateInput(
        research_topic="Benefits of Miele WTI 360"
    )
    summary = graph.invoke(research_input)
    cleaned_summary = re.sub(r'<think>.*?</think>', '', summary['final_summary'], flags=re.DOTALL)
    cleaned_summary = re.sub(r'\n{3,}', '\n\n', cleaned_summary)
    print(cleaned_summary)
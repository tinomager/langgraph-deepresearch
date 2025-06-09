import operator
from dataclasses import dataclass, field
from typing import TypedDict, Annotated 

# summary state data class - core & preserves all important information
@dataclass(kw_only=True)
class SummaryState:
    """Summary state data class."""
    research_topic: str = field(default=None) # report topic
    search_query: str = field(default=None) # search query
    research_results : Annotated[list, operator.add] = field(default_factory=list) # web research results
    sources_gathered : Annotated[list, operator.add] = field(default_factory=list) # sources gathered (urls)
    research_loop_count : int = field(default=0) # research loop count - for iteration tracking
    final_summary: str = field(default=None) # final report

# summary state input object -  to let user define the research topic
dataclass(kw_only=True)
class SummaryStateInput(TypedDict):
    """user input"""
    research_topic: str = field(default=None) # report topic

# summary state output object - to store all the output info
@dataclass(kw_only=True)
class SummaryStateOutput(TypedDict):
    """Summary output"""
    final_summary: str = field(default=None) # Final report
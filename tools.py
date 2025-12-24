"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch  # type: ignore[import-not-found]

from .configuration import Configuration
from .utils import load_chat_model


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


async def simple_answer(query: str) -> Optional[dict[str, Any]]:
    """Answer the user's query if it's a simple question or request.
    This function is designed to handle straightforward queries that can be answered
    without complex processing or external data retrieval.
    """
    Configuration = Configuration.from_context()
    model = load_chat_model()
    response = await model.ainvoke(query)
    return cast(dict[str, Any], {"response": response.content})


TOOLS: List[Callable[..., Any]] = [simple_answer]
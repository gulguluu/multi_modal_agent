#!/usr/bin/env python3
"""
Search MCP Server
Provides web search capabilities through the Model Context Protocol
"""

import logging
from typing import Any, Dict, List

from fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
server = FastMCP("SearchServer")
search_tool = DuckDuckGoSearchRun()


@server.tool()
def search_web(query: str) -> str:
    """
    Search the web for information using DuckDuckGo.

    Args:
        query: The search query

    Returns:
        Search results as text
    """
    try:
        logger.info(f"Searching web for: {query}")
        result = search_tool.run(query)
        return result
    except Exception as e:
        logger.error(f"Error searching web: {str(e)}")
        return f"Error searching web: {str(e)}"


@server.tool()
def search_company_info(company_name: str) -> Dict[str, Any]:
    """
    Search for specific company information.

    Args:
        company_name: Name of the company to search for

    Returns:
        Dictionary with company information
    """
    try:
        logger.info(f"Searching for company info: {company_name}")
        hq_query = f"{company_name} headquarters location"
        hq_result = search_tool.run(hq_query)
        website_query = f"{company_name} official website"
        website_result = search_tool.run(website_query)
        about_query = f"what does {company_name} do business industry"
        about_result = search_tool.run(about_query)
        return {
            "company_name": company_name,
            "headquarters_info": hq_result,
            "website_info": website_result,
            "about_info": about_result,
        }
    except Exception as e:
        logger.error(f"Error searching company info: {str(e)}")
        return {"error": str(e), "company_name": company_name}


@server.resource()
def get_search_capabilities() -> List[str]:
    """Get information about the search capabilities"""
    return [
        "Web search using DuckDuckGo",
        "Company information search",
        "No API key required",
    ]


@server.resource()
def health() -> dict:
    """Health check endpoint for Docker healthchecks"""
    return {
        "status": "healthy",
        "server": "search",
        "capabilities": ["web search", "company information search"],
    }


if __name__ == "__main__":
    logger.info("Starting Search MCP Server...")
    server.run(host="0.0.0.0", port=8002)

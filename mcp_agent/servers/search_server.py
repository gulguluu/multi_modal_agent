#!/usr/bin/env python3
"""
Search MCP Server
Provides web search capabilities through the Model Context Protocol
"""

import json
import logging
from typing import Any, Dict, List

from fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
server = FastMCP("SearchServer", host="0.0.0.0", port=8002)
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
def search_company_info(company_name: str) -> str:
    """
    Search for information about a company.

    Args:
        company_name: Name of the company to search for

    Returns:
        JSON string with company information
    """
    try:
        if "system" in company_name and "assistant" in company_name:
            if "assistant" in company_name:
                parts = company_name.split("assistant")
                if len(parts) > 1:
                    company_name = parts[1].strip()
                    if company_name.startswith(":"):
                        company_name = company_name[1:].strip()
                    if "." in company_name:
                        company_name = company_name.split(".")[0].strip()
        
        if len(company_name) > 50:
            company_name = " ".join(company_name.split()[:3])
        
        logger.info(f"Searching for company: {company_name}")
        
        query = f"{company_name} company information headquarters website"
        
        search_results = search_tool.run(query)
        
        headquarters_info = "No information found"
        website_info = ""
        about_info = ""
        
        if search_results and len(search_results) > 100:
            if "headquarters" in search_results.lower():
                sentences = search_results.split(".")
                for sentence in sentences:
                    if "headquarters" in sentence.lower():
                        headquarters_info = sentence.strip() + "."
                        break
            else:
                headquarters_info = search_results[:200]
            
            if "www." in search_results or "http" in search_results:
                import re
                urls = re.findall(r'(https?://[\w\.-]+|www\.[\w\.-]+)', search_results)
                if urls:
                    website_info = urls[0]
            
            about_info = search_results[:300] if len(search_results) > 300 else search_results
        
        result = {
            "company_name": company_name,
            "headquarters_info": headquarters_info,
            "website_info": website_info,
            "about_info": about_info
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error searching for company info: {str(e)}")
        return json.dumps({"error": str(e), "company_name": company_name})


def get_search_capabilities() -> List[str]:
    """Get information about the search capabilities"""
    return [
        "Web search using DuckDuckGo",
        "Company information search",
        "No API key required",
    ]


@server.resource("data://health")
def health() -> dict:
    """Health check endpoint for Docker healthchecks"""
    return {
        "status": "healthy",
        "server": "search",
        "capabilities": ["web search", "company information search"],
        "hello": "world"
    }


if __name__ == "__main__":
    logger.info("Starting Search MCP Server with FastMCP 2.0.0...")
    server.run("sse")

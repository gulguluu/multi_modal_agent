#!/usr/bin/env python3
"""
Search MCP Server
Provides web search capabilities through the Model Context Protocol
"""

import json
import logging
from typing import List, Dict, Any

from langchain_community.tools import DuckDuckGoSearchRun
from fastmcp import FastMCP

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
    logger.info(f"Searching web for: {query}")
    try:
        search_results = search_tool.invoke(query)
        logger.info(f"Search results length: {len(search_results) if search_results else 0}")
        logger.info(f"Search results preview: {search_results[:200] if search_results else 'None'}...")
        return search_results
    except Exception as e:
        logger.error(f"Error searching web: {str(e)}")
        return "No search results found."


@server.tool()
def search_recipes(ingredients: str) -> str:
    """
    Search for recipes based on provided ingredients.

    Args:
        ingredients: List of ingredients to search recipes for

    Returns:
        JSON string with recipe information
    """
    logger.info(f"Searching for recipes with ingredients: {ingredients}")

    try:
        if ingredients.strip().startswith("[") and ingredients.strip().endswith("]"):
            ingredients_list = json.loads(ingredients)
            if isinstance(ingredients_list, list):
                ingredients = ", ".join(ingredients_list)
    except:
        ingredients = ingredients.strip("[]").replace('"', "").replace("'", "")
    ingredients = ingredients[:200] if len(ingredients) > 200 else ingredients
    if not ingredients or ingredients.lower() in [
        "[]",
        "none",
        "unknown",
        "no ingredients",
    ]:
        return json.dumps(
            {
                "ingredients": [],
                "recipes": [],
                "error": "No ingredients were identified.",
            }
        )

    query = f"recipes with {ingredients} easy homemade"
    logger.info(f"Recipe search query: {query}")
    search_results = search_web(query)
    logger.info(f"Recipe search results received, length: {len(search_results) if search_results else 0}")
    recipes = []
    if search_results and len(search_results) > 100:
        for line in search_results.split("\n"):
            if "recipe" in line.lower() or any(
                food in line.lower() for food in ["dish", "meal", "cook", "bake"]
            ):
                recipes.append(line.strip())
        recipes = recipes[:5] if len(recipes) > 5 else recipes
    result = {
        "ingredients": ingredients,
        "recipes": recipes,
        "full_results": search_results[:1000] if search_results else "",
    }
    return json.dumps(result)


def get_search_capabilities() -> List[str]:
    """Get information about the search capabilities"""
    return [
        "Web search using DuckDuckGo",
        "Recipe search",
        "No API key required",
    ]


if __name__ == "__main__":
    logger.info("Starting Search MCP Server with FastMCP 2.0.0...")
    server.run("sse")

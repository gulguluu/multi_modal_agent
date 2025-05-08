#!/usr/bin/env python3
"""
Search MCP Server
Provides web search capabilities through the Model Context Protocol
"""

import json
import logging
from typing import List, Dict, Any

from duckduckgo_search import DDGS
from fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
server = FastMCP("SearchServer", host="0.0.0.0", port=8002)
# Initialize DuckDuckGo search client
ddgs = DDGS()


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
        # Get search results using the DDGS client directly
        results = ddgs.text(query, max_results=10)
        
        # Format the results as a single text string
        if results:
            formatted_results = "\n\n".join(
                [f"{result['title']}\n{result['href']}\n{result['body']}" for result in results]
            )
            return formatted_results
        else:
            return "No search results found."
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

    # Process ingredients - handle both JSON and text formats
    try:
        # If it looks like a JSON array, convert to comma-separated string
        if ingredients.strip().startswith("[") and ingredients.strip().endswith("]"):
            ingredients_list = json.loads(ingredients)
            if isinstance(ingredients_list, list):
                ingredients = ", ".join(ingredients_list)
    except:
        # Not valid JSON, clean up the string
        ingredients = ingredients.strip("[]").replace('"', "").replace("'", "")

    # Limit length and check for empty ingredients
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

    # Search for recipes
    query = f"recipes with {ingredients} easy homemade"
    search_results = search_web(query)

    # Extract recipe names from search results
    recipes = []
    if search_results and len(search_results) > 100:
        for line in search_results.split("\n"):
            if "recipe" in line.lower() or any(
                food in line.lower() for food in ["dish", "meal", "cook", "bake"]
            ):
                recipes.append(line.strip())
        recipes = recipes[:5] if len(recipes) > 5 else recipes

    # Return formatted results
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

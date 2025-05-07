#!/usr/bin/env python3
"""
Search MCP Server
Provides web search capabilities through the Model Context Protocol
"""

import json
import logging
from typing import List

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
def search_recipes(ingredients: str) -> str:
    """
    Search for recipes based on provided ingredients.

    Args:
        ingredients: List of ingredients to search recipes for

    Returns:
        JSON string with recipe information
    """
    try:
        # Simple ingredient processing - trust the vision model output
        if isinstance(ingredients, str):
            # If it looks like a JSON array, try to parse it
            if ingredients.strip().startswith('[') and ingredients.strip().endswith(']'):
                try:
                    ingredients_list = json.loads(ingredients)
                    if isinstance(ingredients_list, list):
                        ingredients = ", ".join(ingredients_list)
                except json.JSONDecodeError:
                    # Not valid JSON, just use as text
                    logger.info(f"Using raw text from vision model: {ingredients}")
                    # Simple cleanup - remove brackets if they exist
                    ingredients = ingredients.strip('[]').replace('"', '').replace('\'', '')
            
        # Limit length for very long inputs
        if len(ingredients) > 200:
            ingredients = ingredients[:200]
        
        # If no ingredients, just return empty result
        if not ingredients or ingredients.lower() in ["[]", "none", "unknown", "no ingredients"]:
            logger.warning(f"No ingredients found in: {ingredients}")
            return json.dumps({
                "ingredients": [],
                "recipes": [],
                "error": "No ingredients were identified. Please try with a clearer image of food items."
            }, indent=2)
            
        # Construct a search query for recipes
        query = f"recipes with {ingredients} easy homemade"
        
        logger.info(f"Searching for recipes with ingredients: {ingredients}")
        # Use the search_web function to avoid duplicating search logic
        search_results = search_web(query)
        
        # Process search results
        recipes = []
        if search_results and len(search_results) > 100:
            # Extract recipe names
            lines = search_results.split('\n')
            for line in lines:
                if 'recipe' in line.lower() or any(food in line.lower() for food in ['dish', 'meal', 'cook', 'bake']):
                    recipes.append(line.strip())
            
            # Limit to top 5 recipes
            recipes = recipes[:5] if len(recipes) > 5 else recipes
        
        result = {
            "ingredients": ingredients,
            "search_query": query,
            "recipes": recipes,
            "full_results": search_results[:1000] if len(search_results) > 1000 else search_results
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error searching for recipes: {str(e)}")
        return json.dumps({
            "ingredients": ingredients if isinstance(ingredients, str) else str(ingredients),
            "recipes": [],
            "error": f"Error searching for recipes: {str(e)}"
        }, indent=2)


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

#!/usr/bin/env python3
"""
Multi-Modal Agent Client for analyzing food images and suggesting recipes
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
import socket
import sys
from typing import Dict, Any, Optional

import requests
from fastmcp import Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Server configurations - read from environment variables with fallbacks
VISION_SERVER_URL = os.environ.get("VISION_SERVER_URL", "http://localhost:8000")
LLM_SERVER_URL = os.environ.get("LLM_SERVER_URL", "http://localhost:8001")
SEARCH_SERVER_URL = os.environ.get("SEARCH_SERVER_URL", "http://localhost:8002")

logger.info(f"Connecting to Vision Server at: {VISION_SERVER_URL}")
logger.info(f"Connecting to LLM Server at: {LLM_SERVER_URL}")
logger.info(f"Connecting to Search Server at: {SEARCH_SERVER_URL}")


class MultiModalAgent:
    """Multi-Modal Agent for analyzing food images and suggesting recipes"""

    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        self._check_server_connectivity()

    async def suggest_recipe(self, image_path: str, is_url: bool = False) -> Dict[str, Any]:
        """Analyze a food image and suggest recipes

        Args:
            image_path: Path to image file or URL
            is_url: Whether the image_path is a URL

        Returns:
            Dict containing detected food items and recipe suggestions
        """
        local_image_path = self._download_image(image_path, is_url) if is_url else image_path
        food_items = await self._identify_food_items(local_image_path)
        recipe_suggestions = await self._get_recipe_suggestions(food_items)
        
        return {"detected_food_items": food_items, "recipe_suggestions": recipe_suggestions}

    def _check_server_connectivity(self):
        """Check basic connectivity to all servers"""
        logger.info("Checking server connectivity...")
        
        servers = {
            "Vision Server": VISION_SERVER_URL,
            "LLM Server": LLM_SERVER_URL,
            "Search Server": SEARCH_SERVER_URL
        }
        
        for name, url in servers.items():
            try:
                response = requests.get(f"{url}/health", timeout=2)
                if response.status_code == 200:
                    logger.info(f"✅ Connection successful to {name}")
                else:
                    logger.warning(f"⚠️ {name} returned status code {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to {name}: {str(e)}")


    def _download_image(self, image_url: str, is_url: bool) -> str:
        """Download image from URL to local path"""
        logger.info(f"Downloading image from URL: {image_url}")
        
        # Set a proper user agent to avoid being blocked by websites
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Download the image
        response = requests.get(image_url, stream=True, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Create a temporary directory if needed
        temp_dir = Path("/tmp/food_images")
        temp_dir.mkdir(exist_ok=True, parents=True)
        local_path = str(temp_dir / "food_image.jpg")
        
        # Save the image
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Image downloaded to: {local_path}")
        return local_path
    
    def _extract_text_from_response(self, response: Any) -> str:
        """Helper method to extract text content from MCP responses"""
        if hasattr(response, 'content') and response.content:
            for item in response.content:
                if hasattr(item, 'type') and item.type == 'text':
                    return item.text
        elif hasattr(response, 'result'):
            return str(response.result)
        return str(response)
    
    async def _call_mcp_tool(self, server_url: str, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """Call an MCP tool with simple retry logic"""
        logger.info(f"Calling {tool_name} on {server_url}")
        
        # Try the call with retries
        for attempt in range(self.max_retries + 1):
            try:
                # Create client and make the call
                client = Client(f"{server_url}/sse")
                async with client:
                    result = await asyncio.wait_for(
                        client.call_tool(tool_name, params),
                        timeout=60.0
                    )
                    logger.info(f"Received response from {tool_name}")
                    return result
                    
            except (asyncio.TimeoutError, Exception) as e:
                # Handle both timeout and other errors the same way
                logger.warning(f"Error with {tool_name} (attempt {attempt+1}/{self.max_retries+1}): {str(e)}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
        
        logger.error(f"All attempts to call {tool_name} failed")
        return None


    async def _identify_food_items(self, image_path: str) -> str:
        """Identify food items in image using the Vision Server"""
        logger.info(f"Identifying food items in image: {image_path}")
        
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return "[]"
        
        # Call the vision server MCP tool to identify food items
        result = await self._call_mcp_tool(
            VISION_SERVER_URL, 
            "identify_food_items", 
            {"image_path": image_path}
        )
        
        # Extract food items from the response
        food_items_text = self._extract_text_from_response(result)
        
        # Simple deduplication of food items
        if food_items_text:
            try:
                import json
                import re
                
                # Extract items using regex and deduplicate
                matches = re.findall(r'"([^"]+)"', food_items_text)
                if matches:
                    unique_items = sorted(list(set(matches)))
                    food_items_text = json.dumps(unique_items)
                    print(f"✅ Detected food items: {food_items_text}")
                
            except Exception as e:
                logger.warning(f"Error processing food items: {e}")
                
            return food_items_text
        
        return "[]"
    
    async def _get_recipe_suggestions(self, food_items: str) -> str:
        """Get recipe suggestions using Search Server and LLM Server"""
        logger.info(f"Getting recipe suggestions for ingredients: {food_items}")
        
        # Quick check for empty food items
        if food_items == "[]" or not food_items.strip():
            return "No recipe suggestions available as no food items could be identified."
        
        # Step 1: Search for recipes with the identified ingredients
        print("\n==== SEARCHING FOR RECIPES ====\n")
        
        # Call the search server with the food items
        search_result = await self._call_mcp_tool(
            SEARCH_SERVER_URL,
            "search_recipes",
            {"ingredients": food_items}
        )
        
        # Extract search results as a string
        search_result_text = ""
        if search_result:
            try:
                import json
                # Convert to string if it's a dictionary
                if isinstance(search_result, dict):
                    search_result_text = json.dumps(search_result)
                else:
                    search_result_text = self._extract_text_from_response(search_result)
            except Exception as e:
                logger.warning(f"Error processing search results: {e}")
                search_result_text = str(search_result)
        
        # Log search results status
        print(f"Search results found: {len(search_result_text) > 100}")
        
        # Step 2: Generate a recipe based on the ingredients and search results
        print("\n==== GENERATING RECIPE SUGGESTION ====\n")
        
        # Prepare parameters for the LLM server
        llm_params = {
            "ingredients": str(food_items),
            "max_tokens": 1000,
            "search_results": str(search_result_text)[:1000] if search_result_text else 
                              "No specific recipes found for these ingredients."
        }
        
        # Call the LLM server to generate a recipe
        recipe_result = await self._call_mcp_tool(
            LLM_SERVER_URL,
            "generate_recipe",
            llm_params
        )
        
        # Extract and return the recipe text
        if recipe_result:
            return self._extract_text_from_response(recipe_result)
        
        return f"Unable to generate recipe suggestions for {food_items}. Please try again with a clearer image."


async def main():
    """Main function to run the multi-modal agent for recipe suggestions"""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Multi-Modal Agent for Recipe Suggestions")
        parser.add_argument("--image", type=str, required=True, help="Path to food image file or URL")
        parser.add_argument("--url", action="store_true", help="Flag indicating if --image is a URL")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
        args = parser.parse_args()

        # Set debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize agent
        print("🧠 Initializing Multi-Modal Recipe Agent...")
        agent = MultiModalAgent()

        # Process the image
        print(f"🍽️ Analyzing food image {'URL' if args.url else 'file'}: {args.image}")
        
        # Run with timeout
        result = await asyncio.wait_for(
            agent.suggest_recipe(args.image, is_url=args.url),
            timeout=args.timeout
        )
        
        # Display results
        print(f"✅ Detected food items: {result['detected_food_items']}")
        print("")
        print("🍳 Recipe Suggestion:")
        print(f"{result['recipe_suggestions']}")
            
    except asyncio.TimeoutError:
        logger.error(f"Operation timed out after {args.timeout} seconds")
        print(f"\n❌ Error: Operation timed out after {args.timeout} seconds")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(130)

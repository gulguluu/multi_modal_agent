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
import time
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
        self.max_retries = 2
        self.retry_delay = 1  # seconds
        
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
        servers = [
            ("vision-server", 8000),
            ("llm-server", 8001),
            ("search-server", 8002)
        ]
        
        logger.info("Checking server connectivity...")
        
        for hostname, port in servers:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((hostname, port))
                if result == 0:
                    logger.info(f"✅ Connection successful to {hostname}:{port}")
                else:
                    logger.error(f"❌ Connection failed to {hostname}:{port}")
                sock.close()
            except Exception as e:
                logger.error(f"❌ Connection error to {hostname}:{port}: {str(e)}")

    
    def _download_image(self, image_url: str, is_url: bool) -> str:
        """Download image from URL to local path"""
        logger.info(f"Downloading image from URL: {image_url}")
        
        # Set a proper user agent to avoid being blocked by websites
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/'
        }
        
        # Add timeout to avoid hanging on slow connections
        response = requests.get(image_url, stream=True, headers=headers, timeout=10)
        response.raise_for_status()
        
        data_dir = Path("/app/data")
        data_dir.mkdir(exist_ok=True)
        local_path = str(data_dir / "downloaded_logo.png")
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Image downloaded to: {local_path}")
        return local_path
    

    
    async def _call_mcp_tool(self, server_url: str, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """Call an MCP tool with retry logic and proper error handling"""
        logger.info(f"Calling {tool_name} on {server_url}")
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._run_tool_call(server_url, tool_name, params),
                    timeout=60.0  # Increased timeout for larger models
                )
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Tool call timed out (attempt {attempt+1}/{self.max_retries+1})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Error calling {tool_name}: {str(e)} (attempt {attempt+1}/{self.max_retries+1})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
        
        logger.error(f"All attempts to call {tool_name} failed")
        return None
    
    async def _run_tool_call(self, server_url: str, tool_name: str, params: Dict[str, Any]) -> Any:
        """Run a single tool call"""
        client = Client(f"{server_url}/sse")
        
        try:
            async with client:
                result = await client.call_tool(tool_name, params)
                logger.info(f"Received response from {tool_name}")
                return result
        except Exception as e:
            logger.error(f"Error calling {tool_name}: {str(e)}")
            return None


    async def _identify_food_items(self, image_path: str) -> str:
        """Identify food items in image using the specialized Vision Server function"""
        logger.info(f"Identifying food items for recipe suggestions: {image_path}")
        
        # Verify image exists before sending to server
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return "[]"
            
        # Log image details
        try:
            file_size = os.path.getsize(image_path)
            logger.info(f"Image file size: {file_size} bytes")
        except Exception as e:
            logger.warning(f"Could not get image file details: {str(e)}")
        
        # Call vision server with specialized food item identification function
        print("\n==== IDENTIFYING FOOD ITEMS FOR RECIPES ====\n")
        print(f"Processing image: {image_path}")
        
        result = await self._call_mcp_tool(
            VISION_SERVER_URL,
            "identify_food_items",  # Using the specialized food identification function
            {"image_path": image_path}
        )
        
        if result:
            print("\n==== FOOD ITEMS DETECTED BY VISION MODEL ====\n")
            print(f"Raw response: {result}")
            
            # Extract food items from response
            try:
                if hasattr(result, 'content') and result.content:
                    for item in result.content:
                        if hasattr(item, 'type') and item.type == 'text':
                            food_items = item.text if hasattr(item, 'text') else ''
                            print(f"Extracted food items: {food_items}")
                            return food_items
                        elif isinstance(item, dict) and item.get('type') == 'text':
                            food_items = item.get('text', '')
                            print(f"Extracted food items: {food_items}")
                            return food_items
                elif hasattr(result, 'result'):
                    food_items = str(result.result)
                    print(f"Extracted food items: {food_items}")
                    return food_items
                else:
                    food_items = str(result)
                    print(f"Extracted food items: {food_items}")
                    return food_items
            except Exception as e:
                logger.error(f"Error extracting food items from result: {str(e)}")
                if isinstance(result, str):
                    # If result is already a string, just return it
                    return result
        
        # Fallback to general analyze_image if identify_food_items fails
        logger.warning("Specialized food identification failed, trying general image analysis")
        fallback_result = await self._call_mcp_tool(
            VISION_SERVER_URL,
            "analyze_image",
            {"image_path": image_path, "prompt": "List all food items visible in this image. Present as a JSON array of strings."}
        )
        
        if fallback_result:
            try:
                if hasattr(fallback_result, 'content') and fallback_result.content:
                    for item in fallback_result.content:
                        if hasattr(item, 'type') and item.type == 'text':
                            return item.text if hasattr(item, 'text') else '[]'
                elif hasattr(fallback_result, 'result'):
                    return str(fallback_result.result)
                else:
                    return str(fallback_result)
            except Exception as e:
                logger.error(f"Error extracting food items from fallback result: {str(e)}")
        
        logger.warning("Could not identify food items, using empty array")
        return "[]"
    
    async def _get_recipe_suggestions(self, food_items: str) -> str:
        """Get recipe suggestions using Search Server and LLM Server"""
        logger.info(f"Getting recipe suggestions for ingredients: {food_items}")
        
        if food_items == "[]":
            return "No recipe suggestions available as no food items could be identified."
        
        # Step 1: Search for recipes with the identified ingredients
        print("\n==== SEARCHING FOR RECIPES ====\n")
        print(f"Searching for recipes with ingredients: {food_items}")
        
        search_result = await self._call_mcp_tool(
            SEARCH_SERVER_URL,
            "search_recipes",
            {"ingredients": food_items}
        )
        
        # Extract search results - simplified extraction
        search_result_text = ""
        if search_result:
            print(f"Raw search result received")
            
            # Simple extraction logic - trust the model response format
            if hasattr(search_result, 'content'):
                for item in search_result.content:
                    if hasattr(item, 'type') and item.type == 'text':
                        search_result_text = item.text
                        break
            elif hasattr(search_result, 'result'):
                search_result_text = str(search_result.result)
            else:
                search_result_text = str(search_result)
        
        # Step 2: Generate a recipe based on the ingredients and search results
        print("\n==== GENERATING RECIPE SUGGESTION ====\n")
        
        recipe_result = await self._call_mcp_tool(
            LLM_SERVER_URL,
            "generate_recipe",
            {
                "ingredients": food_items,
                "search_results": search_result_text[:2000] if search_result_text else "",
                "max_tokens": 1000
            }
        )
        
        # Extract recipe - simplified extraction
        if recipe_result:
            print("Recipe generated successfully")
            
            # Simple extraction logic
            if hasattr(recipe_result, 'content'):
                for item in recipe_result.content:
                    if hasattr(item, 'type') and item.type == 'text':
                        return item.text
            elif hasattr(recipe_result, 'result'):
                return str(recipe_result.result)
            else:
                return str(recipe_result)
        
        # If we reach here, something went wrong
        return f"Unable to generate recipe suggestions for {food_items}. Please try again with a clearer image."

    




    



async def main():
    """Main function to run the multi-modal agent for recipe suggestions"""
    try:
        parser = argparse.ArgumentParser(description="Multi-Modal Agent for Recipe Suggestions")
        parser.add_argument(
            "--image", 
            type=str, 
            required=True,
            help="Path to food image file or URL"
        )
        parser.add_argument(
            "--url", 
            action="store_true", 
            help="Flag indicating if --image is a URL"
        )
        parser.add_argument(
            "--debug", 
            action="store_true", 
            help="Enable debug logging"
        )
        parser.add_argument(
            "--timeout", 
            type=int, 
            default=120,
            help="Timeout in seconds for the entire operation"
        )
        args = parser.parse_args()

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger("httpx").setLevel(logging.INFO)
            logging.getLogger("mcp").setLevel(logging.DEBUG)

        print("🧠 Initializing Multi-Modal Recipe Agent...")
        agent = MultiModalAgent()

        print(f"🍽️ Analyzing food image {'URL' if args.url else 'file'}: {args.image}")
        
        # Set a timeout for the entire operation
        try:
            # Use asyncio.wait_for to set a timeout for the entire operation
            result = await asyncio.wait_for(
                agent.suggest_recipe(args.image, is_url=args.url),
                timeout=args.timeout
            )
            
            print(f"✅ Detected food items: {result['detected_food_items']}")
            print("")
            print("🍳 Recipe Suggestion:")
            print(f"{result['recipe_suggestions']}")
            
        except asyncio.TimeoutError:
            logger.error(f"Operation timed out after {args.timeout} seconds")
            print(f"\n❌ Error: Operation timed out after {args.timeout} seconds")
            sys.exit(1)

    except asyncio.CancelledError:
        logger.error("Operation was cancelled")
        print("\n❌ Error: Operation was cancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    # Use a more robust asyncio run approach
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

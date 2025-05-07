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
        """Check connectivity to all servers and log results"""
        servers = [
            ("vision-server", 8000),
            ("llm-server", 8001),
            ("search-server", 8002)
        ]
        
        logger.info("Checking server connectivity...")
        
        for hostname, port in servers:
            try:
                ip_address = socket.gethostbyname(hostname)
                logger.info(f"✅ DNS resolution successful for {hostname}: {ip_address}")
            except socket.gaierror:
                logger.error(f"❌ DNS resolution failed for {hostname}")
        
        for hostname, port in servers:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((hostname, port))
                if result == 0:
                    logger.info(f"✅ TCP connection successful to {hostname}:{port}")
                else:
                    logger.error(f"❌ TCP connection failed to {hostname}:{port} (error code: {result})")
                sock.close()
            except Exception as e:
                logger.error(f"❌ TCP connection error to {hostname}:{port}: {str(e)}")
        
        for hostname, port in servers:
            url = f"http://{hostname}:{port}/health"
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    logger.info(f"✅ HTTP request successful to {url}: {response.status_code}")
                else:
                    logger.error(f"❌ HTTP request failed to {url}: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ HTTP request error to {url}: {str(e)}")
    
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
        hostname = server_url.replace('http://', '').split(':')[0]
        port = int(server_url.replace('http://', '').split(':')[1]) if ':' in server_url else 80
        
        logger.info(f"Attempting to connect to {hostname}:{port} for tool: {tool_name}")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((hostname, port))
            if result == 0:
                logger.info(f"✅ TCP connection test successful to {hostname}:{port}")
            else:
                logger.error(f"❌ TCP connection test failed to {hostname}:{port} (error code: {result})")
            sock.close()
        except Exception as e:
            logger.error(f"❌ TCP connection test error to {hostname}:{port}: {str(e)}")
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Connecting to {server_url}/sse (attempt {attempt+1}/{self.max_retries+1})")
                result = await asyncio.wait_for(
                    self._run_tool_call(server_url, tool_name, params),
                    timeout=30.0
                )
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Tool call timed out (attempt {attempt+1}/{self.max_retries+1})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
            except asyncio.CancelledError:
                logger.error(f"Task was cancelled (attempt {attempt+1}/{self.max_retries+1})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Error calling {tool_name}: {str(e)} (attempt {attempt+1}/{self.max_retries+1})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
        
        logger.error(f"All attempts to call {tool_name} failed")
        return None
    
    async def _run_tool_call(self, server_url: str, tool_name: str, params: Dict[str, Any]) -> Any:
        """Run a single tool call without using TaskGroups to avoid cancellation issues"""
        client = Client(f"{server_url}/sse")
        
        try:
            async with client:
                result = await client.call_tool(tool_name, params)
                logger.info(f"Raw response from {tool_name}:")
                logger.info(f"Response type: {type(result)}")
                logger.info(f"Response attributes: {dir(result)}")
                logger.info(f"Response repr: {repr(result)}")
                
                return result
        except Exception as e:
            logger.error(f"Error calling {tool_name}: {str(e)}")
            return None


    async def _identify_food_items(self, image_path: str) -> str:
        """Identify food items in image using Vision Server"""
        logger.info("Analyzing image with Vision Server")
        result = await self._call_mcp_tool(
            VISION_SERVER_URL,
            "analyze_image",
            {"image_path": image_path, "prompt": "List all food items visible in this image. Present as a JSON array of strings."}
        )
        
        if result:
            print("\n==== RAW VISION SERVER RESPONSE ====")
            print(f"Type: {type(result)}")
            if hasattr(result, '__dict__'):
                print(f"Dict representation: {result.__dict__}")
            print(f"String representation: {str(result)}")
            print("===================================\n")
            
            try:
                if hasattr(result, 'content') and result.content:
                    for item in result.content:
                        if hasattr(item, 'type') and item.type == 'text':
                            food_items = item.text if hasattr(item, 'text') else ''
                            if food_items:
                                logger.info(f"Detected food items: {food_items}")
                                return food_items
                        elif isinstance(item, dict) and item.get('type') == 'text':
                            food_items = item.get('text', '')
                            if food_items:
                                logger.info(f"Detected food items: {food_items}")
                                return food_items
                elif hasattr(result, 'result'):
                    food_items = str(result.result)
                    logger.info(f"Detected food items: {food_items}")
                    return food_items
                else:
                    food_items = str(result)
                    logger.info(f"Detected food items: {food_items}")
                    return food_items
            except Exception as e:
                logger.error(f"Error extracting food items from result: {str(e)}")
                logger.debug(f"Raw result: {result}")
        
        logger.warning("Could not identify food items, using fallback")
        return "[]"
    
    async def _get_recipe_suggestions(self, food_items: str) -> str:
        """Get recipe suggestions using Search Server and LLM Server"""
        logger.info(f"Getting recipe suggestions for: {food_items}")
        
        if food_items == "[]":
            return "No recipe suggestions available as no food items could be identified."
        
        # First, use the LLM to generate a search query
        search_query_prompt = f"Given these ingredients: {food_items}, generate a search query to find recipes that use most or all of these ingredients. Return ONLY the search query without any additional text."
        
        search_query_result = await self._call_mcp_tool(
            LLM_SERVER_URL,
            "generate_text",
            {"prompt": search_query_prompt, "max_tokens": 100}
        )
        
        if not search_query_result:
            search_query = f"recipes with {food_items}"
        else:
            try:
                if hasattr(search_query_result, 'content'):
                    for item in search_query_result.content:
                        if hasattr(item, 'type') and item.type == 'text':
                            search_query = item.text
                            break
                    else:
                        search_query = str(search_query_result)
                else:
                    search_query = str(search_query_result)
            except Exception as e:
                logger.error(f"Error extracting search query: {str(e)}")
                search_query = f"recipes with {food_items}"
        
        logger.info(f"Generated search query: {search_query}")
        
        # Now use the search server to find recipes
        result = await self._call_mcp_tool(
            SEARCH_SERVER_URL,
            "search_web",
            {"query": search_query}
        )
        
        if result:
            print("\n==== RAW SEARCH SERVER RESPONSE ====\n")
            print(f"Type: {type(result)}")
            if hasattr(result, '__dict__'):
                print(f"Dict representation: {result.__dict__}")
            print(f"String representation: {str(result)}")
            print("===================================\n")
            
            search_result_text = ""
            try:
                if hasattr(result, 'content'):
                    for item in result.content:
                        if hasattr(item, 'type') and item.type == 'text':
                            search_result_text = item.text
                            break
                    else:
                        search_result_text = str(result)
                elif hasattr(result, 'result'):
                    search_result_text = str(result.result)
                else:
                    search_result_text = str(result)
            except Exception as e:
                logger.error(f"Error extracting search results: {str(e)}")
                search_result_text = str(result)
            
            logger.info(f"Search results: {search_result_text[:200]}...")
            
            # Finally, use the LLM to synthesize a recipe suggestion
            final_prompt = f"""
            Given these ingredients: {food_items}
            And these recipe search results: 
            {search_result_text[:2000]}
            
            Suggest the best recipe the user can make with these ingredients. Format your response as follows:
            1. Recipe name (bold)
            2. Brief description
            3. Ingredients list (bullet points)
            4. Simple step-by-step instructions (numbered)
            5. Cooking time and servings
            
            Only suggest recipes that primarily use the ingredients detected in the image. Be concise and practical.
            """
            print(f"final_prompt: {final_prompt}")
            
            recipe_result = await self._call_mcp_tool(
                LLM_SERVER_URL,
                "generate_text",
                {"prompt": final_prompt, "max_tokens": 1000}
            )
            
            if not recipe_result:
                return f"Unable to generate recipe suggestions for {food_items}."
            
            recipe_text = ""
            try:
                if hasattr(recipe_result, 'content'):
                    for item in recipe_result.content:
                        if hasattr(item, 'type') and item.type == 'text':
                            recipe_text = item.text
                            break
                    else:
                        recipe_text = str(recipe_result)
                elif hasattr(recipe_result, 'result'):
                    recipe_text = str(recipe_result.result)
                else:
                    recipe_text = str(recipe_result)
            except Exception as e:
                logger.error(f"Error extracting recipe: {str(e)}")
                recipe_text = str(recipe_result)
            
            return recipe_text
        else:
            return f"No recipe suggestions found for {food_items}."
    




    



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

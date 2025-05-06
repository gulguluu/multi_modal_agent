#!/usr/bin/env python3
"""
Multi-Modal Agent Client for analyzing logos and retrieving company information
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional

import requests
from fastmcp import Client
import httpx

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
    """Multi-Modal Agent for analyzing logos and retrieving company information"""

    def __init__(self):
        self.max_retries = 2
        self.retry_delay = 1  # seconds

    async def analyze_logo(self, image_path: str, is_url: bool = False) -> Dict[str, Any]:
        """Analyze a logo image and retrieve company information

        Args:
            image_path: Path to image file or URL
            is_url: Whether the image_path is a URL

        Returns:
            Dict containing analysis results and company information
        """
        local_image_path = self._download_image(image_path, is_url) if is_url else image_path
        company_name = await self._identify_logo(local_image_path)
        company_info = await self._get_company_info(company_name)
        
        return {"detected_logo": company_name, "company_info": company_info}
    
    def _download_image(self, image_url: str, is_url: bool) -> str:
        """Download image from URL to local path"""
        logger.info(f"Downloading image from URL: {image_url}")
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        
        data_dir = Path("/app/data")
        data_dir.mkdir(exist_ok=True)
        local_path = str(data_dir / "downloaded_logo.png")
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Image downloaded to: {local_path}")
        return local_path
    
    async def _call_mcp_tool(self, server_url: str, tool_name: str, params: Dict[str, Any]) -> Optional[str]:
        """Call an MCP tool with retry logic and proper error handling"""
        for attempt in range(self.max_retries + 1):
            try:
                # FastMCP 2.2.10 doesn't support timeout parameter
                async with Client(f"{server_url}/sse") as client:
                    # Use a shield to prevent cancellation from propagating
                    result = await asyncio.shield(client.run_tool(tool_name, params))
                    return result
            except asyncio.CancelledError:
                logger.warning(f"MCP call to {tool_name} was cancelled (attempt {attempt+1}/{self.max_retries+1})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"All attempts to call {tool_name} failed due to cancellation")
                    return None
            except Exception as e:
                logger.error(f"Error calling {tool_name}: {str(e)} (attempt {attempt+1}/{self.max_retries+1})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"All attempts to call {tool_name} failed")
                    return None

    async def _identify_logo(self, image_path: str) -> str:
        """Identify logo in image using Vision Server"""
        logger.info("Analyzing image with Vision Server")
        result = await self._call_mcp_tool(
            VISION_SERVER_URL,
            "analyze_image",
            {"image_path": image_path, "prompt": "What company logo is this?"}
        )
        
        if result:
            company_name = result.strip()
            logger.info(f"Detected logo/company: {company_name}")
            return company_name
        else:
            logger.warning("Could not identify logo, using fallback")
            return "Unknown"
    
    async def _get_company_info(self, company_name: str) -> str:
        """Get company information using Search Server"""
        logger.info(f"Getting company info for: {company_name}")
        
        if company_name == "Unknown":
            return "No company information available as the logo could not be identified."
            
        result = await self._call_mcp_tool(
            SEARCH_SERVER_URL,
            "search_company_info",
            {"company_name": company_name}
        )
        
        if result:
            return result if isinstance(result, str) else str(result)
        else:
            return f"Could not retrieve information for {company_name}."


async def main():
    """Main function to run the multi-modal agent"""
    try:
        parser = argparse.ArgumentParser(description="Multi-Modal Agent for Logo Analysis")
        parser.add_argument(
            "--image", 
            type=str, 
            required=True,
            help="Path to logo image file or URL"
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
            default=60,
            help="Timeout in seconds for the entire operation"
        )
        args = parser.parse_args()

        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger("httpx").setLevel(logging.INFO)
            logging.getLogger("mcp").setLevel(logging.DEBUG)

        print("🧠 Initializing Multi-Modal Agent...")
        agent = MultiModalAgent()

        print(f"🖼️ Analyzing image {'URL' if args.url else 'file'}: {args.image}")
        
        # Set a timeout for the entire operation
        try:
            # Use asyncio.wait_for to set a timeout for the entire operation
            result = await asyncio.wait_for(
                agent.analyze_logo(args.image, is_url=args.url),
                timeout=args.timeout
            )
            
            print(f"✅ Detected logo/company name: {result['detected_logo']}")
            print("")
            print("🎯 Company Information:")
            print(f"{result['company_info']}")
            
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
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

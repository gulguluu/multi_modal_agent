#!/usr/bin/env python3
"""
Multi-Modal Agent Client for analyzing logos and retrieving company information
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
    """Multi-Modal Agent for analyzing logos and retrieving company information"""

    def __init__(self):
        self.max_retries = 2
        self.retry_delay = 1  # seconds
        
        self._check_server_connectivity()

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


    async def _identify_logo(self, image_path: str) -> str:
        """Identify logo in image using Vision Server"""
        logger.info("Analyzing image with Vision Server")
        result = await self._call_mcp_tool(
            VISION_SERVER_URL,
            "analyze_image",
            {"image_path": image_path, "prompt": "Identify the company logo in this image. Be specific and concise."}
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
                            company_name = item.text if hasattr(item, 'text') else ''
                            if company_name:
                                logger.info(f"Detected logo/company: {company_name}")
                                return company_name
                        elif isinstance(item, dict) and item.get('type') == 'text':
                            company_name = item.get('text', '')
                            if company_name:
                                logger.info(f"Detected logo/company: {company_name}")
                                return company_name
                elif hasattr(result, 'result'):
                    company_name = str(result.result)
                    logger.info(f"Detected logo/company: {company_name}")
                    return company_name
                else:
                    company_name = str(result)
                    logger.info(f"Detected logo/company: {company_name}")
                    return company_name
            except Exception as e:
                logger.error(f"Error extracting company name from result: {str(e)}")
                logger.debug(f"Raw result: {result}")
        
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
            print("\n==== RAW SEARCH SERVER RESPONSE ====")
            print(f"Type: {type(result)}")
            if hasattr(result, '__dict__'):
                print(f"Dict representation: {result.__dict__}")
            print(f"String representation: {str(result)}")
            print("===================================\n")
            
            try:
                if hasattr(result, 'content') and result.content:
                    for item in result.content:
                        if hasattr(item, 'type') and item.type == 'text':
                            json_text = item.text if hasattr(item, 'text') else f"No information found for {company_name}."
                            try:
                                import json
                                info = json.loads(json_text)
                                return json_text
                            except json.JSONDecodeError:
                                return json_text
                        elif isinstance(item, dict) and item.get('type') == 'text':
                            return item.get('text', f"No information found for {company_name}.")
                elif hasattr(result, 'result'):
                    return str(result.result)
                else:
                    return str(result)
            except Exception as e:
                logger.error(f"Error extracting company info from result: {str(e)}")
                logger.debug(f"Raw result: {result}")
                return f"Error retrieving information for {company_name}."
        else:
            return f"No information found for {company_name}."
    




    



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

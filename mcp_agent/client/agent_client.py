#!/usr/bin/env python3
"""
Multi-Modal Agent Client
Connects to MCP servers for vision, LLM, and search capabilities
Orchestrates the multi-modal agent workflow
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastmcp import Client
from langchain.agents import AgentType, initialize_agent
from langchain.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    """Client for orchestrating multi-modal agent interactions using MCP servers"""

    def __init__(self):
        """Initialize the multi-modal agent client"""
        self.vision_tools = None
        self.llm_tools = None
        self.search_tools = None
        self.all_tools = []
        self.agent = None

    async def connect_to_servers(self, max_retries=5, retry_delay=5):
        """Connect to all MCP servers and load tools
        
        Args:
            max_retries: Maximum number of connection retry attempts
            retry_delay: Delay in seconds between retry attempts
        """
        logger.info("Connecting to MCP servers...")
        
        # Try to connect to servers with retries
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Connection attempt {attempt}/{max_retries}...")
                
                # Try to connect to all servers
                vision_tools = await self._load_tools_from_server(VISION_SERVER_URL, "Vision Server")
                llm_tools = await self._load_tools_from_server(LLM_SERVER_URL, "LLM Server")
                search_tools = await self._load_tools_from_server(SEARCH_SERVER_URL, "Search Server")
                
                # If we got here, we connected successfully
                self.vision_tools = vision_tools
                self.llm_tools = llm_tools
                self.search_tools = search_tools
                
                # Combine all tools
                self.all_tools = []
                if self.vision_tools:
                    self.all_tools.extend(self.vision_tools)
                if self.search_tools:
                    self.all_tools.extend(self.search_tools)
                
                if self.all_tools:
                    logger.info(f"Successfully connected on attempt {attempt}")
                    logger.info(f"Loaded {len(self.all_tools)} tools from MCP servers")
                    return
                
                logger.warning(f"Connected to servers but no tools were loaded. Retrying...")
            except Exception as e:
                logger.warning(f"Connection attempt {attempt} failed: {str(e)}")
            
            # If we've reached the max retries, give up
            if attempt >= max_retries:
                break
                
            # Wait before retrying
            logger.info(f"Waiting {retry_delay} seconds before next attempt...")
            await asyncio.sleep(retry_delay)
        
        # If we get here, we failed to connect after all retries
        raise ValueError(
            "Failed to load any tools from MCP servers after multiple attempts. "
            "Check server status and connectivity."
        )

    async def _load_tools_from_server(
        self, server_url: str, server_name: str
    ) -> List[BaseTool]:
        """Load tools from an MCP server

        Args:
            server_url: URL of the MCP server
            server_name: Name of the server for logging

        Returns:
            List of tools loaded from the server
        """
        try:
            # Explicitly use the /sse endpoint
            sse_url = f"{server_url}/sse"
            logger.info(f"Connecting to {server_name} at {sse_url}")
            
            # Create a new client for each connection attempt
            client = Client(sse_url)
            
            # Use the load_mcp_tools function but with a proper timeout and error handling
            try:
                # Increase timeout to 30 seconds to allow for server startup
                tools = await asyncio.wait_for(load_mcp_tools(client), timeout=30.0)
                
                if tools:
                    logger.info(
                        f"Successfully loaded {len(tools)} tools from {server_name}: {[t.name for t in tools]}"
                    )
                    return tools
                else:
                    logger.warning(f"No tools found in {server_name}")
                    return []
            except Exception as inner_e:
                logger.error(f"Error during tool loading from {server_name}: {str(inner_e)}")
                if hasattr(inner_e, '__cause__') and inner_e.__cause__:
                    logger.error(f"Caused by: {str(inner_e.__cause__)}")
                raise
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to {server_name} after 30 seconds")
            raise ValueError(f"Timeout connecting to {server_name}")
        except Exception as e:
            logger.error(f"Error connecting to {server_name}: {str(e)}")
            # Include the cause if available for better debugging
            if hasattr(e, '__cause__') and e.__cause__:
                logger.error(f"Caused by: {str(e.__cause__)}")
            raise ValueError(f"Failed to connect to {server_name}: {str(e)}")

    async def _get_llm_chain(self):
        """Create a chain using the LLM server's generate_text tool"""
        if not self.llm_tools:
            raise ValueError("LLM tools not loaded. Call connect_to_servers() first.")
        generate_text_tool = None
        for tool in self.llm_tools:
            if tool.name == "generate_text":
                generate_text_tool = tool
                break
        if not generate_text_tool:
            raise ValueError("generate_text tool not found in LLM server")

        return generate_text_tool

    async def _get_prompt_from_server(self):
        """Get the ReAct agent prompt from the LLM server"""
        llm_sse_url = f"{LLM_SERVER_URL}/sse"
        client = Client(llm_sse_url)
        async with client:
            prompts = await asyncio.wait_for(client.list_prompts_mcp(), timeout=10.0)
            prompt_names = [p.name for p in prompts]
            if "react_agent_prompt" in prompt_names:
                prompt_result = await asyncio.wait_for(
                    client.get_prompt("react_agent_prompt"), timeout=10.0
                )
                prompt_text = prompt_result.text
                return PromptTemplate.from_template(prompt_text)
            else:
                raise ValueError("react_agent_prompt not found on the LLM server")

    async def create_agent(self):
        """Create the multi-modal agent using tools from all servers"""
        if not self.all_tools:
            raise ValueError("Tools not loaded. Call connect_to_servers() first.")
        logger.info("Creating multi-modal agent...")
        llm_chain = await self._get_llm_chain()
        prompt = await self._get_prompt_from_server()
        self.agent = initialize_agent(
            tools=self.all_tools,
            llm=llm_chain,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            agent_kwargs={"prompt": prompt},
        )

        logger.info("Multi-modal agent created successfully")
        return self.agent

    async def analyze_logo(
        self, image_path: str, is_url: bool = False
    ) -> Dict[str, Any]:
        """Analyze a logo image and retrieve company information

        Args:
            image_path: Path to the logo image file or URL
            is_url: Flag indicating if image_path is a URL

        Returns:
            Dict containing analysis results and company information
        """
        try:
            vision_tool = None
            for tool in self.vision_tools:
                if tool.name == "analyze_image":
                    vision_tool = tool
                    break

            if not vision_tool:
                raise ValueError("analyze_image tool not found in vision server")

            local_image_path = image_path
            if is_url:
                logger.info(f"Downloading image from URL: {image_path}")
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                data_dir = Path("/app/data")
                data_dir.mkdir(exist_ok=True)
                local_image_path = str(data_dir / "downloaded_logo.png")
                with open(local_image_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Image downloaded to: {local_image_path}")
            result = await vision_tool.ainvoke(
                {"image_path": local_image_path, "prompt": "What company logo is this?"}
            )
            company_name = result.strip()
            logger.info(f"Detected logo/company: {company_name}")
            query = f"Provide detailed information about {company_name}. Include headquarters location, what they do, and their website."
            response = await self.agent.ainvoke(query)
            return {"detected_logo": result, "company_info": response["output"]}
        except Exception as e:
            logger.error(f"Error in analyze_logo: {str(e)}")
            return {"error": str(e), "detected_logo": None, "company_info": None}


async def main():
    """Main function to run the multi-modal agent"""
    try:
        parser = argparse.ArgumentParser(description="Multi-Modal Agent for Logo Analysis")
        parser.add_argument(
            "--image", 
            type=str, 
            default="./logo1.png",
            help="Path to local image file or URL of logo image"
        )
        parser.add_argument(
            "--url", 
            action="store_true",
            help="Flag to indicate the image parameter is a URL"
        )
        parser.add_argument(
            "--debug", 
            action="store_true",
            help="Enable debug mode with more detailed logging"
        )
        args = parser.parse_args()
        
        # Set debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger('mcp').setLevel(logging.DEBUG)
        
        image_path = args.image
        is_url = args.url

        print("🧠 Initializing Multi-Modal Agent...")
        agent = MultiModalAgent()

        print("🔌 Connecting to MCP servers...")
        try:
            # Use longer retry delay and more retries for initial startup
            await agent.connect_to_servers(max_retries=10, retry_delay=10)
        except Exception as conn_err:
            logger.error(f"Connection error: {str(conn_err)}")
            print(f"❌ Connection error: {str(conn_err)}")
            # Print more detailed error information
            if hasattr(conn_err, '__cause__') and conn_err.__cause__:
                logger.error(f"Caused by: {str(conn_err.__cause__)}")
                print(f"  Caused by: {str(conn_err.__cause__)}")
            return
        
        if not agent.all_tools:
            print("❌ Error: No tools loaded from MCP servers. Check server status and network connectivity.")
            return

        print(f"🖼️ Analyzing {'image URL' if is_url else 'image file'}: {image_path}")
        try:
            result = await agent.analyze_logo(image_path, is_url=is_url)
            
            if "error" in result and result["error"]:
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ Detected logo/company name:", result["detected_logo"])
                print("\n🎯 Final Answer:\n", result["company_info"])
        except Exception as analyze_err:
            logger.error(f"Error analyzing logo: {str(analyze_err)}")
            print(f"❌ Error analyzing logo: {str(analyze_err)}")
            # Print more detailed error information
            if hasattr(analyze_err, '__cause__') and analyze_err.__cause__:
                logger.error(f"Caused by: {str(analyze_err.__cause__)}")
                print(f"  Caused by: {str(analyze_err.__cause__)}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"❌ Error: {str(e)}")
        # Print more detailed error information
        if hasattr(e, '__cause__') and e.__cause__:
            logger.error(f"Caused by: {str(e.__cause__)}")
            print(f"  Caused by: {str(e.__cause__)}")
        # Print traceback for debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        if args.debug:
            print("\nTraceback:")
            print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())

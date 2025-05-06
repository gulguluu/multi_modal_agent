#!/usr/bin/env python3
"""
Multi-Modal Agent Client
Connects to MCP servers for vision, LLM, and search capabilities
Orchestrates the multi-modal agent workflow
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from fastmcp import Client
from fastmcp.transports.http import HttpTransport
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

    async def connect_to_servers(self):
        """Connect to all MCP servers and load tools"""
        logger.info("Connecting to MCP servers...")

        # Connect to vision server
        self.vision_tools = await self._load_tools_from_server(
            VISION_SERVER_URL, "Vision Server"
        )

        # Connect to LLM server
        self.llm_tools = await self._load_tools_from_server(LLM_SERVER_URL, "LLM Server")

        # Connect to search server
        self.search_tools = await self._load_tools_from_server(
            SEARCH_SERVER_URL, "Search Server"
        )

        # Combine all tools
        self.all_tools = []
        if self.vision_tools:
            self.all_tools.extend(self.vision_tools)
        if self.search_tools:
            self.all_tools.extend(self.search_tools)

        logger.info(f"Loaded {len(self.all_tools)} tools from MCP servers")

    async def _load_tools_from_server(
        self, server_url: str, server_name: str
    ) -> List[BaseTool]:
        """Load tools from a specific MCP server

        Args:
            server_url: URL of the MCP server
            server_name: Name of the server for logging

        Returns:
            List of tools loaded from the server
        """
        try:
            client = Client(server_url)
            async with client:
                tools = await load_mcp_tools(client)
                logger.info(
                    f"Loaded {len(tools)} tools from {server_name}: {[t.name for t in tools]}"
                )
                return tools
        except Exception as e:
            logger.error(f"Error connecting to {server_name}: {str(e)}")
            return []

    async def _get_llm_chain(self):
        """Create a chain using the LLM server's generate_text tool"""
        if not self.llm_tools:
            raise ValueError("LLM tools not loaded. Call connect_to_servers() first.")

        # Find the generate_text tool
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
        try:
            client = Client(LLM_SERVER_URL)
            async with client:
                prompts = await client.list_prompts_mcp()
                prompt_names = [p.name for p in prompts]

                if "react_agent_prompt" in prompt_names:
                    prompt_result = await client.get_prompt("react_agent_prompt")
                    prompt_text = prompt_result.text
                    return PromptTemplate.from_template(prompt_text)

                # Fallback to default prompt
                return PromptTemplate.from_template(
                        """
                    You are an intelligent agent that helps identify companies from logos and find detailed company information.
                    
                    Tools available:
                    {tools}
                    
                    Use this format:
                    
                    Question: the question to solve
                    Thought: your reasoning step
                    Action: tool to use, one of [{tool_names}]
                    Action Input: the input for the tool
                    Observation: the result
                    ... (repeat Thought/Action/Action Input/Observation)
                    Thought: I now know the final answer
                    Final Answer: the complete answer with company name, HQ, website, services
                    
                    Begin!
                    
                    Question: {input}
                    """
                    )
        except Exception as e:
            logger.error(f"Error getting prompt from server: {str(e)}")
            # Fallback to default prompt
            return PromptTemplate.from_template(
                """
            You are an intelligent agent that helps identify companies from logos and find detailed company information.
            
            Tools available:
            {tools}
            
            Use this format:
            
            Question: the question to solve
            Thought: your reasoning step
            Action: tool to use, one of [{tool_names}]
            Action Input: the input for the tool
            Observation: the result
            ... (repeat Thought/Action/Action Input/Observation)
            Thought: I now know the final answer
            Final Answer: the complete answer with company name, HQ, website, services
            
            Begin!
            
            Question: {input}
            """
            )

    async def create_agent(self):
        """Create the multi-modal agent using tools from all servers"""
        if not self.all_tools:
            raise ValueError("Tools not loaded. Call connect_to_servers() first.")

        logger.info("Creating multi-modal agent...")

        # Get the LLM chain
        llm_chain = await self._get_llm_chain()

        # Get the prompt
        prompt = await self._get_prompt_from_server()

        # Initialize agent
        self.agent = initialize_agent(
            tools=self.all_tools,
            llm=llm_chain,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )

        logger.info("Multi-modal agent created successfully")
        return self.agent

    async def analyze_logo(self, image_path: str) -> Dict[str, Any]:
        """Analyze a logo image and retrieve company information

        Args:
            image_path: Path to the logo image file

        Returns:
            Dict containing analysis results and company information
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            logger.info(f"Analyzing image: {image_path}")

            # Connect to servers if not already connected
            if not self.all_tools:
                await self.connect_to_servers()

            # Find the analyze_image tool
            analyze_image_tool = None
            for tool in self.vision_tools:
                if tool.name == "analyze_image":
                    analyze_image_tool = tool
                    break

            if not analyze_image_tool:
                raise ValueError("analyze_image tool not found in vision server")

            # Analyze the image
            result = await analyze_image_tool.ainvoke({"image_path": image_path})
            logger.info(f"Detected logo/company: {result}")

            # Create the agent if not already created
            if not self.agent:
                await self.create_agent()

            # Generate query for company information
            query = f"Identify the company shown in the image as '{result}' and give me its details like name, headquarters, website, and what it does."

            logger.info("Running agent to gather company information")
            response = await self.agent.ainvoke(query)

            return {"detected_logo": result, "company_info": response["output"]}
        except Exception as e:
            logger.error(f"Error in analyze_logo: {str(e)}")
            return {"error": str(e), "detected_logo": None, "company_info": None}


async def main():
    """Main function to run the multi-modal agent"""
    try:
        image_path = "./logo1.png"

        print("🧠 Initializing Multi-Modal Agent...")
        agent = MultiModalAgent()

        print("🔌 Connecting to MCP servers...")
        await agent.connect_to_servers()

        print("🖼️ Analyzing image...")
        result = await agent.analyze_logo(image_path)

        if "error" in result and result["error"]:
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Detected logo/company name:", result["detected_logo"])
            print("\n🎯 Final Answer:\n", result["company_info"])

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())

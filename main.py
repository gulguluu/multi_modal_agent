#!/usr/bin/env python3
# Multi-Modal Agent for Logo Recognition and Company Information Retrieval
# Optimized for Intel GPUs

import logging
import os
from typing import Any, Dict, Optional

import intel_extension_for_pytorch as ipex
import torch
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """Analyzes images to identify logos and content using a vision-language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = None,
    ):
        """Initialize the image analyzer with the specified model.

        Args:
            model_name: HuggingFace model identifier
            device: Computing device (xpu, cuda, cpu). If None, uses xpu if available, else cpu.
        """
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=torch.float16
            )
            self.device = device or ("xpu" if torch.xpu.is_available() else "cpu")
            logger.info(f"Using device: {self.device} for image analysis")
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error initializing ImageAnalyzer: {str(e)}")
            raise

    def analyze(
        self, image_path: str, prompt: str = "Identify the logo shown in this image?"
    ) -> str:
        """Analyze an image to identify its content.

        Args:
            image_path: Path to the image file
            prompt: Text prompt for the vision model

        Returns:
            Text description of the image content
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[image], return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_length=250)

            return self.processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"


def image_tool_fn(image_path: str) -> str:
    """Tool function to analyze an image and identify logos/content.

    Args:
        image_path: Path to the image file

    Returns:
        Description of the identified logo/content
    """
    try:
        analyzer = ImageAnalyzer()
        return analyzer.analyze(image_path)
    except Exception as e:
        logger.error(f"Error in image_tool_fn: {str(e)}")
        return f"Error analyzing image: {str(e)}"


def load_local_llm():
    """Load and configure a local language model optimized for Intel hardware.

    Returns:
        HuggingFacePipeline: Configured language model pipeline
    """
    try:
        model_id = "HuggingFaceH4/zephyr-7b-beta"
        logger.info(f"Loading language model: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        device = "xpu" if torch.xpu.is_available() else "cpu"
        logger.info(f"Using device: {device} for language model")
        model = model.to(device)
        if device == "xpu":
            model = ipex.optimize(model, dtype=torch.float16)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
        )

        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:
        logger.error(f"Error loading language model: {str(e)}")
        raise


def create_custom_prompt():
    """Create a prompt template for the agent's interaction format.

    Returns:
        PromptTemplate: Configured prompt template
    """
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


def create_agent():
    """Create and configure the multi-modal agent with tools and LLM.

    Returns:
        Agent: Configured LangChain agent
    """
    try:
        image_tool = Tool(
            name="ImageAnalyzer",
            func=image_tool_fn,
            description="Analyze an image and identify the company/logo shown.",
        )

        search_tool = Tool(
            name="duckduckgo_search",
            func=DuckDuckGoSearchRun(),
            description="Search for company details like HQ, website, services etc.",
        )

        tools = [image_tool, search_tool]
        llm = load_local_llm()
        prompt = create_custom_prompt()
        chain = prompt | llm

        return initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            llm_chain=chain,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=5,
        )
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise


def analyze_logo(image_path: str) -> Dict[str, Any]:
    """Analyze a logo image and retrieve company information.

    Args:
        image_path: Path to the logo image file

    Returns:
        Dict containing analysis results and company information
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Analyzing image: {image_path}")
        result = image_tool_fn(image_path)
        logger.info(f"Detected logo/company: {result}")

        query = f"Identify the company shown in the image as '{result}' and give me its details like name, headquarters, website, and what it does."

        agent = create_agent()
        logger.info("Running agent to gather company information")
        response = agent.invoke(query)

        return {"detected_logo": result, "company_info": response["output"]}
    except Exception as e:
        logger.error(f"Error in analyze_logo: {str(e)}")
        return {"error": str(e), "detected_logo": None, "company_info": None}


if __name__ == "__main__":
    try:
        image_path = "./logo1.png"

        print("🧠 Analyzing image...")
        result = image_tool_fn(image_path)
        print("✅ Detected logo/company name:", result)

        query = f"Identify the company shown in the image as '{result}' and give me its details like name, headquarters, website, and what it does."

        agent = create_agent()
        print("\n🌐 Running agent...")
        response = agent.invoke(query)

        print("\n🎯 Final Answer:\n", response["output"])
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")

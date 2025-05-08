#!/usr/bin/env python3
# Multi-Modal Agent for Logo Recognition and Company Information Retrieval
# Optimized for Intel GPUs using Moondream2 model

import argparse
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import intel_extension_for_pytorch as ipex
import requests
import torch
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.WARNING, 
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("pyvips").setLevel(logging.ERROR)
logging.getLogger("_logger.py").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

class ImageAnalyzer:
    """Analyzes images to identify logos and content using the Moondream2 vision-language model."""

    def __init__(
        self,
        model_name: str = "vikhyatk/moondream2",
        device: Optional[str] = None,
    ):
        """Initialize the image analyzer with the Moondream2 model.

        Args:
            model_name: HuggingFace model identifier
            device: Computing device (xpu, cuda, cpu). If None, uses xpu if available, else cpu.
        """
        try:
            self.device = device or ("xpu" if torch.xpu.is_available() else "cpu")
            self.dtype = torch.float16 if self.device != "cpu" else torch.bfloat16
            logger.info(
                f"Using device: {self.device} for image analysis with {model_name}"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=self.dtype,
            ).to(self.device)
            self.model.eval()
            if self.device == "xpu":
                logger.info("Applying IPEX optimizations to vision model")
                try:
                    self.model = ipex.optimize(
                        self.model, dtype=torch.float16, inplace=True
                    )
                    logger.info("IPEX optimization applied successfully.")
                except Exception as e:
                    logger.error(f"IPEX optimization failed: {e}", exc_info=True)
                    logger.warning("Proceeding without IPEX optimization due to error.")

        except Exception as e:
            logger.error(f"Error initializing ImageAnalyzer: {str(e)}")
            raise

    def analyze(
        self,
        image_path: str,
        prompt: str = "Identify the company logo shown in this image. Respond with ONLY the company name.",
    ) -> str:
        """Analyze an image to identify its content using Moondream2 model.

        Args:
            image_path: Path to the image file
            prompt: Text prompt for the vision model

        Returns:
            Text description of the image content
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return "Error: Image not found"
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                return f"Error loading image: {str(e)}"
            with torch.inference_mode():
                answer_dict = self.model.query(image=image, question=prompt)
                response = answer_dict["answer"]
            logger.info(f"Raw analysis result: {response}")
            result = response.strip()
            if "\n" in result:
                result = result.split("\n")[0].strip()
            prefixes_to_remove = [
                "The logo is ",
                "This is the logo of ",
                "The company is ",
                "This is a ",
                "I can see the logo of ",
                "The image shows ",
                "The logo belongs to ",
                "This logo is from ",
            ]
            for prefix in prefixes_to_remove:
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix) :].strip()
            if (
                result.lower() == "none"
                or not result
                or result.lower() == "i don't know"
                or result.lower() == "unknown"
            ):
                result = "Unknown Logo"
                logger.warning(
                    "Vision model failed to identify the logo, using fallback: Unknown Logo"
                )

            logger.info(f"Final analysis result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"


def download_image(url: str) -> str:
    """Download an image from a URL and save it to a temporary file.

    Args:
        url: URL of the image to download

    Returns:
        Path to the downloaded image file
    """
    try:
        logger.info(f"Downloading image from URL: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Image downloaded to: {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise


def image_tool_fn(image_path_or_url: str) -> str:
    """Tool function to analyze an image and identify logos/content.

    Args:
        image_path_or_url: Path to the image file or URL

    Returns:
        Description of the identified logo/content
    """
    try:
        if image_path_or_url.startswith("http"):
            image_path = download_image(image_path_or_url)
        else:
            image_path = image_path_or_url
        analyzer = ImageAnalyzer()
        result = analyzer.analyze(image_path)
        if image_path_or_url.startswith("http") and os.path.exists(image_path):
            try:
                os.remove(image_path)
                logger.info(f"Removed temporary file: {image_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")

        return result
    except Exception as e:
        logger.error(f"Error in image_tool_fn: {str(e)}")
        return f"Error analyzing image: {str(e)}"


def load_local_llm():
    """Load and configure a local language model optimized for Intel hardware.

    Returns:
        HuggingFacePipeline: Configured language model pipeline
    """
    try:
        model_id = "Qwen/Qwen2.5-3B-Instruct"
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
            verbose=False,
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
        # Parse command-line arguments
        parser = argparse.ArgumentParser(
            description="Multi-Modal Agent for Logo Recognition"
        )
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--image", type=str, help="Path to the logo image file")
        group.add_argument("--url", type=str, help="URL of the logo image")
        parser.add_argument("--debug", action="store_true", help="Enable debug output")
        parser.add_argument("--quiet", action="store_true", help="Minimal output")
        args = parser.parse_args()
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            for handler in logging.getLogger().handlers:
                handler.setLevel(logging.DEBUG)
            warnings.filterwarnings("default")
        image_source = args.url if args.url else args.image
        if not args.quiet:
            print("\n" + "=" * 60)
            print("🔍 MULTI-MODAL LOGO RECOGNITION AGENT")
            print("=" * 60)
        if not args.quiet:
            print("\n🧠 Analyzing image...")
            if len(image_source) > 60:
                display_source = image_source[:57] + "..."
            else:
                display_source = image_source
            print(f"   Source: {display_source}")
        result = image_tool_fn(image_source)
        if not args.quiet:
            print(f"\n✅ DETECTED: {result}")
        query = f"Identify the company shown in the image as '{result}' and give me its details like name, headquarters, website, and what it does."
        if not args.quiet:
            print("\n🌐 Retrieving company information...")
        agent = create_agent()
        response = agent.invoke(query)
        if not args.quiet:
            print("\n📋 COMPANY PROFILE:")
            print("=" * 60)
        output = response["output"]
        if "Final Answer:" in output:
            output = output.split("Final Answer:")[1].strip()
        print(output)
        if not args.quiet:
            print("\n" + "=" * 60)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()

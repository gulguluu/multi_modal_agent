#!/usr/bin/env python3
"""
Vision Model MCP Server
Provides image analysis capabilities through the Model Context Protocol
Optimized for Intel GPUs using Intel Extension for PyTorch
"""

import logging
import os
from typing import Optional

import intel_extension_for_pytorch as ipex
import torch
from fastmcp import FastMCP
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


server = FastMCP("VisionServer", host="0.0.0.0", port=8000)

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
            logger.info(f"Loading vision model: {model_name}")
            # Ensure model is downloaded properly with progress bar
            logger.info("Downloading and preparing model files...")
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                #cache_dir="/app/.cache/huggingface",
            )
            logger.info("Processor loaded, now loading model...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                #cache_dir="/app/.cache/huggingface",
            )
            self.device = device or ("xpu" if torch.xpu.is_available() else "cpu")
            logger.info(f"Using device: {self.device} for vision model")
            self.model.to(self.device)

            if self.device == "xpu":
                logger.info("Applying IPEX optimizations to vision model")
                self.model = ipex.optimize(self.model, dtype=torch.bfloat16)

        except Exception as e:
            logger.error(f"Error initializing ImageAnalyzer: {str(e)}")
            raise

    def analyze(
        self, image_path: str, prompt: str = "List all food items visible in this image. Present as a JSON array of strings."
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

            logger.info(f"Analyzing image: {image_path}")
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "system",
                    "content": "You are a food recognition expert. "
                               "Identify all food items in the image and list them as a JSON array of strings. "
                               "For example, if shown food items, respond with ['apple', 'banana', 'milk']. "
                               "Be specific and concise. Only include food ingredients that can be used in recipes. "
                               "Do not include prepared dishes, only the raw ingredients. "
                               "If you're uncertain about an item, include it with your best guess."
                },
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
                outputs = self.model.generate(**inputs, max_new_tokens=500)

            result = self.processor.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            
            if "assistant" in result.lower():
                parts = result.split("assistant")
                if len(parts) > 1:
                    result = parts[1].strip()
            result = result.strip()
            if result.startswith(":"):
                result = result[1:].strip()
            if "." in result:
                result = result.split(".")[0].strip()
            if "," in result:
                result = result.split(",")[0].strip()
            
            if result.lower() == "none" or not result or result.lower() == "i don't know" or result.lower() == "unknown":
                result = "[]"
                logger.warning("Vision model failed to identify the food items, using fallback: empty array")
            
            logger.info(f"Analysis result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"


analyzer = ImageAnalyzer()


@server.tool()
def analyze_image(
    image_path: str, prompt: str = "Identify the food items shown in this image?"
) -> str:
    """
    Analyze an image to identify food items.

    Args:
        image_path: Path to the image file
        prompt: Text prompt for the vision model

    Returns:
        Text description of the image content
    """
    return analyzer.analyze(image_path, prompt)


def get_model_info() -> dict:
    """Get information about the loaded vision model"""
    return {
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "device": analyzer.device,
        "capabilities": ["food item identification", "image content analysis"],
    }



if __name__ == "__main__":
    logger.info("Starting Vision Model MCP Server with FastMCP 2.0.0...")
    server.run("sse")

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
    """Analyzes images to identify food items and ingredients using a vision-language model."""

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
                    "content": "You are a professional food recognition expert specializing in recipe ingredients. "
                               "Your task is to identify all food items in the image and list them as a JSON array of strings. "
                               "For example, if shown food items, respond with ['apple', 'banana', 'milk']. "
                               "Be specific and concise. Focus on identifying individual ingredients rather than prepared dishes. "
                               "Include common cooking ingredients like vegetables, fruits, meats, dairy, spices, and pantry staples. "
                               "If you see prepared food, list the main visible ingredients instead of the dish name. "
                               "For packaged foods, identify the main ingredient (e.g., 'pasta' not 'pasta box'). "
                               "If you're uncertain about an item, include your best guess. "
                               "IMPORTANT: Always respond with a valid JSON array, even if empty ([])."
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
    image_path: str, prompt: str = "List all food items visible in this image. Present as a JSON array of strings."
) -> str:
    """
    Analyze an image to identify food items and ingredients for recipes.

    Args:
        image_path: Path to the image file
        prompt: Text prompt for the vision model (default optimized for food detection)

    Returns:
        JSON array of identified food items as a string
    """
    logger.info(f"Analyzing image for food items: {image_path}")
    return analyzer.analyze(image_path, prompt)


@server.tool()
def identify_food_items(image_path: str) -> str:
    """
    Specialized function to identify food items and ingredients in an image for recipe suggestions.
    
    This function is optimized specifically for food detection with a carefully crafted prompt.

    Args:
        image_path: Path to the image file containing food items

    Returns:
        JSON array of identified food items as a string
    """
    logger.info(f"Identifying food items for recipes: {image_path}")
    
    # Verify image exists
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return "[]"
        
    # Log image details
    try:
        file_size = os.path.getsize(image_path)
        logger.info(f"Image file size: {file_size} bytes")
    except Exception as e:
        logger.warning(f"Could not get image file details: {str(e)}")
    
    # Use a specialized prompt for food detection
    specialized_prompt = """Carefully examine this image and identify ALL food items and ingredients visible.
    Focus on individual ingredients that could be used in recipes, not prepared dishes.
    Return ONLY a JSON array of strings with the identified items. Example: ["tomato", "onion", "chicken"]
    Be specific about types when possible (e.g., "red bell pepper" rather than just "pepper").
    If you see packaged foods, name the main ingredient, not the packaging.
    If you see prepared dishes, list the visible ingredients instead.
    If no food items are visible, return an empty array: []."""
    
    # Use the analyzer with the specialized prompt
    result = analyzer.analyze(image_path, specialized_prompt)
    
    # Simple JSON validation with minimal processing
    try:
        import json
        import re
        
        # Check if the result already looks like a JSON array
        if result.strip().startswith('[') and result.strip().endswith(']'):
            try:
                # Try to parse as JSON
                json.loads(result)
                # If it parses successfully, keep it as is
            except json.JSONDecodeError:
                # If it looks like JSON but isn't valid, do minimal cleanup
                logger.info(f"Basic cleanup of JSON-like result: {result[:50]}...")
                # Try to extract a valid JSON array pattern
                json_array_match = re.search(r'\[.*?\]', result)
                if json_array_match:
                    result = json_array_match.group(0)
        else:
            # If it's not in JSON format, wrap it in an array
            logger.info(f"Result not in JSON format, returning as-is: {result}...")
            # The search and LLM components can handle non-JSON text
            
    except Exception as e:
        logger.error(f"Error processing food items result: {str(e)}")
        result = "[]"
    
    logger.info(f"Identified food items: {result}")
    return result


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

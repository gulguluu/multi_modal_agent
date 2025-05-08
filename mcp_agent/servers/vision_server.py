#!/usr/bin/env python3
"""
Vision MCP Server for analyzing images and identifying food items
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union

import intel_extension_for_pytorch as ipex
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Server configuration
PORT = int(os.environ.get("VISION_SERVER_PORT", 8000))
HOST = os.environ.get("VISION_SERVER_HOST", "0.0.0.0")
MODEL_NAME = os.environ.get("VISION_MODEL_NAME", "vikhyatk/moondream2")

server = FastMCP("VisionServer", host=HOST, port=PORT)

class ImageAnalyzer:
    """Analyzes images to identify food items and ingredients using a vision-language model."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
    ):
        """Initialize the image analyzer with the specified model.

        Args:
            model_name: HuggingFace model identifier
            device: Computing device (xpu, cuda, cpu). If None, uses xpu if available, else cpu.
        """
        try:
            logger.info(f"Loading vision model: {model_name}")
            logger.info("Downloading and preparing model files...")
            
            # Determine device and data type
            self.device = device or ("xpu" if torch.xpu.is_available() else "cpu")
            self.dtype = torch.float16 if self.device != "cpu" else torch.float32
            logger.info(f"Using device: {self.device} for vision model")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=self.dtype,
            ).to(self.device)
            self.model.eval()
            
            # Apply IPEX optimizations if using XPU
            if self.device == "xpu":
                logger.info("Applying IPEX optimizations to vision model")
                try:
                    self.model = ipex.optimize(self.model, dtype=torch.float16, inplace=True)
                    logger.info("IPEX optimization applied successfully.")
                except Exception as e:
                    logger.error(f"IPEX optimization failed: {e}", exc_info=True)
                    logger.warning("Proceeding without IPEX optimization due to error.")

        except Exception as e:
            logger.error(f"Error initializing ImageAnalyzer: {str(e)}")
            raise

    def identify_food_items(self, image_path: str) -> str:
        """Identify food items in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            JSON string containing identified food items
        """
        # Check if image exists
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return "[]"
            
        # Load image
        logger.info(f"Analyzing image: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return "[]"
        
        # Prepare food detection query
        user_query = "List out all the food items in this image in a JSON array format. Only include food ingredients that could be used in recipes. Return only the JSON array."
        
        # Process image with model
        try:
            with torch.inference_mode():
                answer_dict = self.model.query(image=image, question=user_query)
                response = answer_dict["answer"]
                
            logger.info(f"Analysis result: {response}")
            
            # Clean up the response if needed
            # If response is not already a JSON array, try to extract it
            if not response.strip().startswith("["):
                import re
                import json
                # Look for array pattern
                array_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if array_match:
                    try:
                        # Validate it's proper JSON
                        food_items = json.loads(array_match.group(0))
                        # Deduplicate food items if it's a list
                        if isinstance(food_items, list):
                            food_items = sorted(list(set(food_items)))
                        response = json.dumps(food_items)
                    except:
                        response = "[]"
                else:
                    response = "[]"
            
            logger.info(f"Identified food items: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return "[]"

    def analyze(self, image_path: str, prompt: str = "List all food items visible in this image. Present as a JSON array of strings.") -> str:
        """Analyze an image to identify its content.
        
        This is a general-purpose image analysis method.
        For food detection specifically, use identify_food_items.

        Args:
            image_path: Path to the image file
            prompt: Text prompt for the vision model

        Returns:
            Text description of the image content
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return "Error: Image not found"
                
            # Load image
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                return f"Error loading image: {str(e)}"
            
            # Process image with model
            with torch.inference_mode():
                answer_dict = self.model.query(image=image, question=prompt)
                response = answer_dict["answer"]
                
            logger.info(f"Analysis result: {response}")
            return response
            
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
    """
    logger.info(f"Identifying food items for recipes: {image_path}")
    result = analyzer.identify_food_items(image_path)
    
    # Print the raw result for debugging
    print(f"\n==== RAW RESULT FROM FOOD ITEM DETECTION ====\n{result}\n==== END RAW RESULT ====\n")
    return result


def get_model_info() -> dict:
    """Get information about the loaded vision model"""
    return {
        "model_name": "vikhyatk/moondream2",
        "device": analyzer.device,
        "capabilities": ["food item identification", "image content analysis"],
    }



if __name__ == "__main__":
    logger.info("Starting Vision Model MCP Server with FastMCP 2.0.0...")
    server.run("sse")

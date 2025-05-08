#!/usr/bin/env python3
"""
LLM MCP Server
Provides text generation capabilities through the Model Context Protocol
Optimized for Intel GPUs using Intel Extension for PyTorch
"""

import logging
from typing import Any, Dict, Optional

import intel_extension_for_pytorch as ipex
import torch
from fastmcp import FastMCP
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
server = FastMCP("LLMServer", host="0.0.0.0", port=8001)


class LLMProcessor:
    """Handles text generation using a local LLM optimized for Intel hardware."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
    ):
        """Initialize the LLM processor with the specified model.

        Args:
            model_id: HuggingFace model identifier
            device: Computing device (xpu, cuda, cpu). If None, uses xpu if available, else cpu.
        """
        try:
            logger.info(f"Loading language model: {model_id}")
            self.model_id = model_id
            
            # Ensure model is downloaded properly with progress bar
            logger.info("Downloading and preparing tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                use_fast=True,
                trust_remote_code=True,
                #cache_dir="/app/.cache/huggingface",
            )
            
            logger.info("Tokenizer loaded, now loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                trust_remote_code=True,
                #cache_dir="/app/.cache/huggingface",
            )

            self.device = device or ("xpu" if torch.xpu.is_available() else "cpu")
            logger.info(f"Using device: {self.device} for language model")
            self.model = self.model.to(self.device)
            if self.device == "xpu":
                logger.info("Applying IPEX optimizations to language model")
                self.model = ipex.optimize(self.model, dtype=torch.float16)

            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1,
            )

        except Exception as e:
            logger.error(f"Error initializing LLMProcessor: {str(e)}")
            raise

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text based on the provided prompt.

        Args:
            prompt: The text prompt to generate from
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        try:
            logger.info(f"Generating text for prompt: {prompt[:50]}...")
            result = self.pipe(prompt, max_new_tokens=max_tokens)
            generated_text = result[0]["generated_text"]
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt) :].strip()
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error generating text: {str(e)}"


llm_processor = LLMProcessor()


@server.tool()
def generate_text(prompt: str, max_tokens: int = 256) -> str:
    """
    Generate text using the local LLM.

    Args:
        prompt: The text prompt to generate from
        max_tokens: Maximum number of tokens to generate

    Returns:
        Generated text response
    """
    try:
        return llm_processor.generate(prompt, max_tokens)
    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        return f"Error generating text: {str(e)}"


@server.tool()
def generate_recipe(ingredients: str, search_results: str = "", max_tokens: int = 512) -> str:
    """
    Generate a recipe based on provided ingredients and optional search results.

    Args:
        ingredients: List of ingredients available for the recipe
        search_results: Optional search results about recipes with these ingredients
        max_tokens: Maximum number of tokens to generate

    Returns:
        Formatted recipe with name, ingredients, instructions, and cooking time
    """
    logger.info(f"Generating recipe for ingredients: {ingredients}")
    
    # Clean up ingredients if it's a JSON array
    if ingredients.strip().startswith('[') and ingredients.strip().endswith(']'):
        try:
            import json
            ingredients_list = json.loads(ingredients)
            if isinstance(ingredients_list, list):
                ingredients = ", ".join(ingredients_list)
        except:
            # Not valid JSON, use as is
            pass
    
    # Build the prompt based on available information
    prompt_parts = [f"Based on these ingredients: {ingredients}"]
    
    # Add search results if available
    if search_results and len(search_results) > 10:
        prompt_parts.append(f"And these recipe search results:\n{search_results[:1000]}")
    
    # Add formatting instructions
    prompt_parts.append("""
    Create a delicious recipe that primarily uses these ingredients. Format your response as follows:
    1. Recipe name (bold)
    2. Brief description
    3. Ingredients list (bullet points)
    4. Simple step-by-step instructions (numbered)
    5. Cooking time and servings
    
    Only suggest recipes that primarily use the ingredients listed. Be concise and practical.
    """)
    
    # Join all parts with newlines
    prompt = "\n".join(prompt_parts)
    
    try:
        return llm_processor.generate(prompt, max_tokens)
    except Exception as e:
        logger.error(f"Error generating recipe: {str(e)}")
        return f"Could not generate a recipe with the provided ingredients. Please try again with different ingredients."


def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded language model"""
    return {
        "model_name": llm_processor.model_id,
        "device": llm_processor.device,
        "max_tokens": 512,
        "temperature": 0.2,
    }


if __name__ == "__main__":
    logger.info("Starting LLM MCP Server with FastMCP 2.0.0...")
    server.run("sse")

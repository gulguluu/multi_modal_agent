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


server = FastMCP("LLMServer")


class LLMProcessor:
    """Handles text generation using a local LLM optimized for Intel hardware."""

    def __init__(
        self,
        model_id: str = "HuggingFaceH4/zephyr-7b-beta",
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
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16
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
                max_new_tokens=256,  # Reduced from 512 for better performance
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
    return llm_processor.generate(prompt, max_tokens)


@server.tool()
def answer_question(question: str) -> str:
    """
    Answer a question using the local LLM.

    Args:
        question: The question to answer

    Returns:
        Answer to the question
    """
    prompt = f"Please answer the following question accurately and concisely:\n\nQuestion: {question}\n\nAnswer:"
    return llm_processor.generate(prompt, max_tokens=128)  # Using fewer tokens for concise answers


def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded language model"""
    return {
        "model_name": llm_processor.model_id,
        "device": llm_processor.device,
        "max_tokens": 512,
        "temperature": 0.2,
    }


def react_agent_prompt() -> str:
    """Prompt template for ReAct agent reasoning"""
    return """
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


@server.resource("data://health")
def health() -> dict:
    """Health check endpoint for Docker healthchecks"""
    model_loaded = hasattr(llm_processor, "model") and llm_processor.model is not None
    test_message = (
        "Model loaded and ready" if model_loaded else "Model not loaded properly"
    )
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "server": "llm",
        "model": llm_processor.model_id,
        "device": llm_processor.device,
        "test": test_message,
        "hello": "world",
    }


if __name__ == "__main__":
    # Use SSE transport as it's more widely supported
    logger.info("Starting LLM MCP Server with FastMCP 2.0.0...")
    # In FastMCP 2.0.0, the run method doesn't take host/port as separate arguments
    server.run("0.0.0.0", 8001, transport="sse")

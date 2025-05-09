# Multi-Modal Agent for Logo Recognition and Recipe generation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Intel XPU](https://img.shields.io/badge/Intel-XPU-0071C5)

This repository contains implementations of a multi-modal agent for logo recognition and company information retrieval, also an example of recipe generator using multi modal agents and mcp, optimized for Intel GPUs using Intel Extension for PyTorch.

## Repository Structure

This repository contains two implementations of the same functionality:

1. **Monolithic Implementation** (root directory): A single-file implementation that combines vision and language models in a simple, straightforward approach.

2. **MCP-Based Implementation** (`mcp_agent` directory): A modern, microservices-based implementation using the Model Context Protocol (MCP) with separate servers for vision, language, and search capabilities.

## Monolithic Implementation

The monolithic implementation (`main.py`) provides a simple, all-in-one solution for logo recognition and company information retrieval:

- Single Python file with all components
- Uses Moondream2 for efficient logo detection
- Easier to understand for beginners
- Runs all models in the same process
- Optimized for Intel GPUs using Intel Extension for PyTorch

### Running the Monolithic Implementation

#### Local Execution

```bash
pip install -r requirements.txt

# Run with a local image file
python main.py --image path/to/logo.png

# Run with an image URL
python main.py --url https://example.com/logo.png
```

#### Docker Execution

```bash
# Build the Docker image
docker build -f Dockerfile.monolithic -t multi-modal-agent:monolithic .

# Run with a local image file (must be in the data directory)
docker run --privileged -v $(pwd)/data:/app/data multi-modal-agent:monolithic --image /app/data/logo.png

# Run with an image URL
docker run --privileged multi-modal-agent:monolithic --url https://example.com/logo.png
```

### About the Moondream2 Model

The monolithic implementation uses the Moondream2 vision-language model, which offers several advantages for logo detection:

- Smaller model size (1.5B parameters) compared to alternatives
- Efficient inference on Intel GPUs
- Good performance on visual recognition tasks
- Optimized with Intel Extension for PyTorch (IPEX)

## MCP-Based Implementation

For a more modern, maintainable, and scalable approach, check out the MCP-based implementation in the `mcp_agent` directory. This implementation:

- Separates concerns into microservices
- Uses the Model Context Protocol (MCP) for standardized communication
- Provides better resource isolation and scaling
- Includes Docker configurations for production deployment
- Implements the ReAct agent pattern with perception capabilities

See the [MCP Agent README](mcp_agent/README.md) for detailed information about this implementation.

## Comparison

| Feature | Monolithic | MCP-Based |
|---------|------------|-----------|
| Architecture | Single process | Microservices |
| Complexity | Low | Medium |
| Scalability | Limited | High |
| Maintainability | Good for small projects | Excellent for large projects |
| Resource Isolation | None | Complete |
| Deployment | Simple | More complex but flexible |
| Intel GPU Support | Yes | Yes |

## Requirements

- Python 3.12+
- Intel GPU with XPU support
- Intel Extension for PyTorch

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Intel for the PyTorch extension and XPU support
- The open-source AI community for models and tools

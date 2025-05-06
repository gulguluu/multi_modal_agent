# Docker Deployment for Multi-Modal Agent with MCP

This directory contains Docker configurations for deploying the multi-modal agent as separate microservices. Each component (vision model, LLM, search) runs in its own container, optimized for its specific requirements.

## Architecture

The deployment consists of four containers:

1. **Vision Server** (`mcp-vision-server`):
   - Uses Intel Extension for PyTorch with XPU acceleration
   - Runs the Qwen2.5-VL-3B-Instruct model for image analysis
   - Exposes port 8000

2. **LLM Server** (`mcp-llm-server`):
   - Uses Intel Extension for PyTorch with XPU acceleration
   - Runs the Zephyr-7b-beta model for text generation
   - Exposes port 8001

3. **Search Server** (`mcp-search-server`):
   - Lightweight Python container
   - Provides web search capabilities using DuckDuckGo
   - Exposes port 8002

4. **Client** (`mcp-client`):
   - Orchestrates the multi-modal agent workflow
   - Connects to all three servers
   - Processes images and provides company information

## Prerequisites

- Docker and Docker Compose
- Intel GPU with appropriate drivers
- Intel Extension for PyTorch container support

## Quick Start

### Building and Running

1. Make sure you're in the `docker` directory:
   ```bash
   cd /Users/runnikri/Coding/multi_modal_agent/mcp_agent/docker
   ```

2. Make the build script executable:
   ```bash
   chmod +x build-and-run.sh
   ```

3. Run the build script:
   ```bash
   ./build-and-run.sh
   ```

This will:
- Build all container images
- Start all services
- Check container health

### Using the Client

To analyze a logo image:

1. Place your image in the `data` directory:
   ```bash
   cp your-logo.png ../data/logo1.png
   ```

2. Run the client:
   ```bash
   docker-compose run client
   ```

### Monitoring

- View logs from all containers:
  ```bash
  docker-compose logs -f
  ```

- View logs from a specific container:
  ```bash
  docker-compose logs -f vision-server
  ```

- Check container health:
  ```bash
  docker-compose ps
  ```

## Container Details

### Vision Server

- Base Image: `intel/intel-extension-for-pytorch:2.7.10-serving-xpu`
- Optimized for Intel GPUs
- Environment variables:
  - `IPEX_OPTIMIZE=1`: Enable IPEX optimizations
  - `IPEX_MERGE_FUSION=1`: Enable operation fusion
  - `IPEX_AUTO_TUNE=1`: Enable auto-tuning
  - `IPEX_VERBOSE=1`: Enable verbose logging

### LLM Server

- Base Image: `intel/intel-extension-for-pytorch:2.7.10-serving-xpu`
- Optimized for Intel GPUs
- Environment variables:
  - `IPEX_OPTIMIZE=1`: Enable IPEX optimizations
  - `IPEX_MERGE_FUSION=1`: Enable operation fusion
  - `IPEX_AUTO_TUNE=1`: Enable auto-tuning
  - `TRANSFORMERS_CACHE=/app/models`: Cache models in volume

### Search Server

- Base Image: `python:3.10-slim`
- Lightweight container without GPU requirements
- Minimal dependencies for web search functionality

### Client

- Base Image: `python:3.10-slim`
- Environment variables:
  - `VISION_SERVER_URL`: URL of the vision server
  - `LLM_SERVER_URL`: URL of the LLM server
  - `SEARCH_SERVER_URL`: URL of the search server

## Volumes

The deployment uses Docker volumes for persistent storage:

- `vision-cache`: Cache for vision model
- `llm-cache`: Cache for LLM
- `llm-models`: Storage for downloaded models
- `../data`: Mounted directory for input/output files

## Customization

### Using Different Models

To use different models, modify the respective Dockerfiles and update the model IDs in the server code.

### Scaling

For production deployments, consider:
- Using Kubernetes for orchestration
- Setting resource limits based on your hardware
- Implementing proper monitoring and logging
- Adding authentication for the MCP servers

## Troubleshooting

### GPU Issues

If the containers can't access the Intel GPU:

1. Check that the Intel GPU is properly recognized:
   ```bash
   lspci | grep -i vga | grep -i intel
   ```

2. Verify that the Intel GPU drivers are installed:
   ```bash
   dpkg -l | grep intel-gpu
   ```

3. Ensure the container has GPU access permissions:
   ```bash
   docker run --privileged intel/intel-extension-for-pytorch:2.7.10-serving-xpu python -c "import torch; print(torch.xpu.is_available())"
   ```

### Network Issues

If containers can't communicate:

1. Check that all containers are running:
   ```bash
   docker-compose ps
   ```

2. Verify network connectivity:
   ```bash
   docker exec -it mcp-client ping vision-server
   ```

3. Check server logs for binding issues:
   ```bash
   docker-compose logs vision-server
   ```

## Cleanup

To stop and remove all containers:

```bash
docker-compose down
```

To also remove volumes:

```bash
docker-compose down -v
```

#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Building and deploying MCP Multi-Modal Agent ===${NC}"
mkdir -p ../data
echo -e "${YELLOW}Checking for Intel GPU availability...${NC}"
if [ -z "$(lspci | grep -i display | grep -i intel)" ]; then
  echo -e "${RED}Warning: Intel GPU not detected. The vision and LLM servers may fall back to CPU.${NC}"
else
  echo -e "${GREEN}Intel GPU detected.${NC}"
fi

echo -e "${YELLOW}Building Docker images...${NC}"
docker-compose build

echo -e "${YELLOW}Starting containers...${NC}"
docker-compose up -d
echo -e "${YELLOW}Checking container status...${NC}"
sleep 5
docker-compose ps

echo -e "${GREEN}=== Deployment complete ===${NC}"
echo -e "${BLUE}To view logs:${NC} docker-compose logs -f"
echo -e "${BLUE}To stop:${NC} docker-compose down"
echo -e "${BLUE}To run the client:${NC} docker-compose run client"

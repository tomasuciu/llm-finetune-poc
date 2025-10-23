#!/bin/bash

set -e

REGISTRY="cr.eu-north1.nebius.cloud/e00v67hgmwbybk0xrr"
IMAGE_NAME="llm-finetune-poc"

# Output colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building and Pushing Docker Image ===${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine project root
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
else
    # Script is in project root
    PROJECT_ROOT="$SCRIPT_DIR"
fi

cd "$PROJECT_ROOT"
echo -e "${GREEN}Working directory: ${PROJECT_ROOT}${NC}"

# Check if Dockerfile exists
if [[ ! -f "Dockerfile" ]]; then
    echo -e "${RED}Error: Dockerfile not found in ${PROJECT_ROOT}${NC}"
    exit 1
fi

# Check if Nebius CLI is installed
if ! command -v nebius &> /dev/null; then
    echo -e "${RED}Error: Nebius CLI is not installed${NC}"
    echo -e "${YELLOW}Install it from: https://docs.nebius.com/cli/${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Configure Nebius CLI credential helper
echo -e "${GREEN}Configuring Nebius CLI as Docker credential helper...${NC}"
if ! nebius registry configure-helper; then
    echo -e "${RED}Error: Failed to configure Nebius CLI credential helper${NC}"
    exit 1
fi

# Test authentication with Nebius registry
echo -e "${GREEN}Testing Nebius registry authentication...${NC}"
if ! docker login ${REGISTRY%%/*} 2>&1 | grep -q "Login Succeeded\|Already logged in"; then
    # Try a more direct test - attempt to list repositories
    if ! nebius container registry list 2>&1 | grep -q "id:"; then
        echo -e "${RED}Error: Unable to authenticate with Nebius Cloud${NC}"
        echo -e "${YELLOW}Please check your Nebius CLI configuration:${NC}"
        echo -e "  1. Run: nebius init"
        echo -e "  2. Ensure you have valid credentials"
        echo -e "  3. Verify your access to the registry: ${REGISTRY}"
        exit 1
    fi
fi
echo -e "${GREEN}Authentication successful${NC}"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

# Get the current git commit hash (short version)
GIT_HASH=$(git rev-parse --short HEAD)
echo -e "${YELLOW}Git commit hash: ${GIT_HASH}${NC}"

# Check if there are uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}Warning: You have uncommitted changes. Consider committing them first.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build the Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:${GIT_HASH} .

# Tag with latest
echo -e "${GREEN}Tagging image as latest...${NC}"
docker tag ${IMAGE_NAME}:${GIT_HASH} ${IMAGE_NAME}:latest

# Tag for Nebius registry with git hash
echo -e "${GREEN}Tagging for Nebius registry...${NC}"
docker tag ${IMAGE_NAME}:${GIT_HASH} ${REGISTRY}/${IMAGE_NAME}:${GIT_HASH}
docker tag ${IMAGE_NAME}:${GIT_HASH} ${REGISTRY}/${IMAGE_NAME}:latest

# Push both tags to Nebius registry
echo -e "${GREEN}Pushing image with git hash tag...${NC}"
if ! docker push ${REGISTRY}/${IMAGE_NAME}:${GIT_HASH}; then
    echo -e "${RED}Error: Failed to push image to registry${NC}"
    echo -e "${YELLOW}This usually means authentication failed or you don't have push access${NC}"
    exit 1
fi

echo -e "${GREEN}Pushing image with latest tag...${NC}"
if ! docker push ${REGISTRY}/${IMAGE_NAME}:latest; then
    echo -e "${RED}Error: Failed to push 'latest' tag to registry${NC}"
    exit 1
fi

echo -e "${GREEN}=== Successfully pushed image ===${NC}"
echo -e "Git hash tag: ${REGISTRY}/${IMAGE_NAME}:${GIT_HASH}"
echo -e "Latest tag:   ${REGISTRY}/${IMAGE_NAME}:latest"

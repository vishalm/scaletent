#!/bin/bash

# ASCII Art and Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Error handling
set -e  # Exit on error
trap 'handle_error $? $LINENO $BASH_LINENO "$BASH_COMMAND" $(printf "::%s" ${FUNCNAME[@]:-})' ERR

# Error handler function
handle_error() {
    local exit_code=$1
    local line_no=$2
    local bash_lineno=$3
    local last_command=$4
    local func_trace=$5
    echo -e "\n${RED}Error occurred in script at line $line_no${NC}"
    echo -e "${RED}Command: $last_command${NC}"
    echo -e "${RED}Exit code: $exit_code${NC}"
    exit $exit_code
}

echo -e "${BLUE}"
cat << "EOF"
   _____ ____    __    __     _________ _____ _   _ _______ 
  / ____/ ___|  /  \  /  \   |__   __|| ____| \ | |__   __|
 | (___| |     / /\ \/ /\ \     | |   | |__ |  \| |  | |   
  \___ \ |    / /  \__/  \ \    | |   |  __|| . ` |  | |   
  ____) ||   / /          \ \   | |   | |___| |\  |  | |   
 |_____/|___/_/            \_\  |_|   |_____|_| \_|  |_|   
                                                            
EOF
echo -e "${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to display status
show_status() {
    local exit_status=$?
    if [ $exit_status -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
    else
        echo -e "${RED}✗ $1${NC}"
        return $exit_status
    fi
}

# Function to check system requirements
check_requirements() {
    echo -e "\n${YELLOW}Checking system requirements...${NC}"
    
    # Check if running on Apple Silicon
    if [[ $(uname -m) != "arm64" ]]; then
        echo -e "${RED}Error: This script is designed for Apple Silicon Macs (M1/M2/M3).${NC}"
        exit 1
    fi
    show_status "Running on Apple Silicon"

    # Check for Docker
    if ! command_exists docker; then
        echo -e "${RED}Error: Docker is not installed. Please install Docker Desktop for Mac first.${NC}"
        echo "Download from: https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    show_status "Docker is installed"

    # Check if Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker daemon is not running. Please start Docker Desktop.${NC}"
        exit 1
    fi
    show_status "Docker daemon is running"

    # Check for docker-compose
    if ! command_exists docker-compose; then
        echo -e "${RED}Error: docker-compose is not installed.${NC}"
        exit 1
    fi
    show_status "docker-compose is installed"
}

# Function to check Docker configuration
check_docker_configuration() {
    echo -e "\n${YELLOW}Checking Docker configuration...${NC}"
    
    # Check Docker platform
    if ! docker info | grep -q "linux/arm64"; then
        echo -e "${YELLOW}Warning: Docker platform linux/arm64 not explicitly enabled${NC}"
        echo -e "Consider enabling linux/arm64 platform in Docker Desktop settings"
    fi
    show_status "Docker platform check completed"
    
    # Check Docker memory allocation
    DOCKER_MEMORY=$(docker info | grep "Total Memory" | awk '{print $3}' | sed 's/GiB//')
    if (( $(echo "$DOCKER_MEMORY < 4" | bc -l) )); then
        echo -e "${YELLOW}Warning: Docker has less than 4GB memory allocated${NC}"
        echo -e "Consider increasing memory allocation in Docker Desktop settings"
    fi
    show_status "Docker memory check completed"
    
    # Verify Docker experimental features
    if ! docker info | grep -q "Experimental: true"; then
        echo -e "${YELLOW}Warning: Docker experimental features not enabled${NC}"
        echo -e "Consider enabling experimental features in Docker Desktop settings"
    fi
    show_status "Docker experimental features check completed"
}

# Function to clean up existing containers and images
cleanup() {
    echo -e "\n${YELLOW}Cleaning up existing containers...${NC}"
    docker-compose -f docker-compose.apple-silicon.yml down --remove-orphans 2>/dev/null || true
    show_status "Cleaned up existing containers"
}

# Function to create necessary directories
setup_directories() {
    echo -e "\n${YELLOW}Setting up directories...${NC}"
    mkdir -p data/models data/db logs config
    show_status "Created necessary directories"
}

# Function to check and copy configuration files
setup_config() {
    echo -e "\n${YELLOW}Setting up configuration...${NC}"
    if [ ! -f "config/config.yaml" ]; then
        if [ -f "config/config.example.yaml" ]; then
            cp config/config.example.yaml config/config.yaml
            show_status "Created config.yaml from example"
        else
            echo -e "${RED}Error: config.example.yaml not found${NC}"
            exit 1
        fi
    else
        show_status "Configuration files already exist"
    fi
}

# Function to build and start containers
start_services() {
    echo -e "\n${YELLOW}Building and starting services...${NC}"
    docker-compose -f docker-compose.apple-silicon.yml build
    show_status "Built Docker images"
    
    echo -e "\n${YELLOW}Starting services...${NC}"
    docker-compose -f docker-compose.apple-silicon.yml up -d
    show_status "Started all services"
}

# Function to check service health
check_services() {
    echo -e "\n${YELLOW}Checking service health...${NC}"
    sleep 5  # Give services time to start
    
    local running_containers=$(docker-compose -f docker-compose.apple-silicon.yml ps -q | wc -l)
    if [ $running_containers -gt 0 ]; then
        show_status "Services are running ($running_containers containers)"
        
        # Check individual service health
        echo -e "\n${YELLOW}Checking individual services...${NC}"
        docker-compose -f docker-compose.apple-silicon.yml ps --services | while read service; do
            if docker-compose -f docker-compose.apple-silicon.yml ps $service | grep -q "Up"; then
                show_status "$service is healthy"
            else
                echo -e "${RED}✗ $service failed to start${NC}"
                return 1
            fi
        done
    else
        echo -e "${RED}Error: Services failed to start${NC}"
        echo -e "${YELLOW}Checking logs for errors...${NC}"
        docker-compose -f docker-compose.apple-silicon.yml logs
        exit 1
    fi
}

# Function to display service information
show_info() {
    echo -e "\n${GREEN}🚀 ScaleTent is running!${NC}"
    echo -e "\n${BLUE}Access the services at:${NC}"
    echo -e "📊 Web Interface: ${GREEN}http://localhost:8501${NC}"
    echo -e "🔧 API: ${GREEN}http://localhost:8000${NC}"
    echo -e "💾 MongoDB: ${GREEN}mongodb://localhost:27017${NC}"
    echo -e "📡 Redis: ${GREEN}redis://localhost:6379${NC}"
    
    echo -e "\n${BLUE}Useful commands:${NC}"
    echo -e "📝 View logs: ${YELLOW}docker-compose -f docker-compose.apple-silicon.yml logs -f${NC}"
    echo -e "🛑 Stop services: ${YELLOW}docker-compose -f docker-compose.apple-silicon.yml down${NC}"
    echo -e "🔄 Restart services: ${YELLOW}docker-compose -f docker-compose.apple-silicon.yml restart${NC}"
    echo -e "🔍 Check status: ${YELLOW}docker-compose -f docker-compose.apple-silicon.yml ps${NC}"
    
    echo -e "\n${BLUE}Docker Configuration Tips:${NC}"
    echo -e "1. Ensure at least 4GB memory is allocated to Docker"
    echo -e "2. Enable linux/arm64 platform in Docker Desktop settings"
    echo -e "3. Enable VirtioFS for better performance"
    echo -e "4. Check Docker Desktop → Settings → Features in development"
}

# Main execution
main() {
    check_requirements
    check_docker_configuration
    cleanup
    setup_directories
    setup_config
    start_services
    check_services
    show_info
}

# Execute main function
main 
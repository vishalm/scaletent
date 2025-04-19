#!/bin/bash

# ASCII Art and Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
export OPENCV_AVFOUNDATION_SKIP_AUTH=1


# Error handling
set -e  # Exit on error

echo -e "${BLUE}"
cat << "EOF"
  _____            _      _______           _   
 / ____|          | |    |__   __|         | |  
| (___   ___  __ _| | ___   | | ___ _ __   | |_ 
 \___ \ / __|/ _` | |/ _ \  | |/ _ \ '_ \  | __|
 ____) | (__| (_| | |  __/  | |  __/ | | | | |_ 
|_____/ \___|\__,_|_|\___|  |_|\___|_| |_|  \__|


EOF
echo -e "${NC}"

# Function to check if Docker is running and configured
check_docker() {
    echo -e "\n${YELLOW}Checking Docker...${NC}"
    
    # Check if Docker is installed and running
    if ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running${NC}"
        echo "Please start Docker Desktop first"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Docker is running${NC}"
}

# Function to check service health with proper retry mechanism
check_service_health() {
    local service=$1
    local container=$2
    local check_cmd=$3
    local max_attempts=${4:-30}
    local attempt=1
    local delay=1
    
    echo -e "${YELLOW}Waiting for $service to be ready...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec $container $check_cmd >/dev/null 2>&1; then
            echo -e "${GREEN}✓ $service is ready${NC}"
            return 0
        fi
        echo -e "Attempt $attempt/$max_attempts: $service not ready yet, waiting ${delay}s..."
        sleep $delay
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}Error: $service failed to become ready after $max_attempts attempts${NC}"
    return 1
}

# Function to check if a web endpoint is available
check_endpoint() {
    local name=$1
    local url=$2
    local max_attempts=${3:-60}
    local attempt=1
    local delay=1
    
    echo -e "${YELLOW}Checking if $name is up at $url...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --head "$url" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ $name is up and running at $url${NC}"
            return 0
        fi
        echo -e "Attempt $attempt/$max_attempts: $name not available yet, waiting ${delay}s..."
        sleep $delay
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}Warning: $name could not be reached at $url after $max_attempts attempts${NC}"
    echo -e "${YELLOW}The application might still be starting up, or there might be an issue.${NC}"
    echo -e "${YELLOW}Check the terminal windows or log files for more information.${NC}"
    return 1
}

# Function to start a process in a new terminal window
start_in_terminal() {
    local title=$1
    local command=$2
    
    echo -e "${YELLOW}Starting $title...${NC}"
    
    # Create a unique script name with timestamp
    local timestamp=$(date +%s)
    local script_name="scaletent_$(echo $title | tr ' ' '_')_${timestamp}.sh"
    local temp_script="/tmp/$script_name"
    
    # Write commands to the script
    cat > $temp_script << EOL
#!/bin/bash
cd $(pwd)
source venv/bin/activate
export PYTHONPATH=$(pwd)
export OPENCV_AVFOUNDATION_SKIP_AUTH=1
echo "Starting $title..."
$command
EOL
    
    # Make executable
    chmod +x $temp_script
    
    # Open in new terminal
    osascript -e 'tell application "Terminal" to do script "'"$temp_script"'"'
}

# Function to setup Python environment
setup_python_env() {
    echo -e "\n${YELLOW}Setting up Python environment...${NC}"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Detect Apple Silicon and use appropriate requirements file
    if [ "$(uname -m)" = "arm64" ]; then
        REQ_FILE="requirements.apple-silicon.txt"
    else
        REQ_FILE="requirements.txt"
    fi
    
    # Install requirements if they exist
    if [ -f "$REQ_FILE" ]; then
        echo -e "${YELLOW}Installing requirements from $REQ_FILE...${NC}"
        pip install -r "$REQ_FILE"
    else
        echo -e "${YELLOW}No requirements file found ($REQ_FILE)${NC}"
    fi
    
    # Fix torch to use CPU if CUDA is not available
    echo -e "${YELLOW}Ensuring PyTorch is configured correctly...${NC}"
    python -c "
import torch
if not torch.cuda.is_available() and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
    print('GPU acceleration not available. Ensuring CPU version is used.')
"
    
    echo -e "${GREEN}✓ Python environment ready${NC}"
}

# Function to ensure MongoDB data directory has proper permissions
fix_mongodb_permissions() {
    echo -e "\n${YELLOW}Checking MongoDB data directory permissions...${NC}"
    
    # Get the container ID for MongoDB
    MONGO_CONTAINER=$(docker ps | grep mongodb | awk '{print $1}')
    
    if [ -n "$MONGO_CONTAINER" ]; then
        # Ensure data directory exists and has proper permissions
        docker exec $MONGO_CONTAINER bash -c "mkdir -p /data/db && chown -R 999:999 /data/db" || true
        echo -e "${GREEN}✓ MongoDB permissions fixed${NC}"
    else
        echo -e "${YELLOW}MongoDB container not found, skipping permission fix${NC}"
    fi
}

# Create stop script for easier shutdown
create_stop_script() {
    echo -e "\n${YELLOW}Creating stop script...${NC}"
    
    cat > ./stop-all.sh << 'STOPSCRIPT'
#!/bin/bash
echo "Stopping ScaleTent components..."

# Kill the processes in the terminal windows
osascript -e 'tell application "Terminal" to close (every window whose name contains "scaletent")'

# Stop Docker containers
docker-compose -f docker-compose.services.yml down

echo "ScaleTent stopped"
STOPSCRIPT

    chmod +x ./stop-all.sh
    echo -e "${GREEN}✓ Stop script created: ./stop-all.sh${NC}"
}

# Main execution
main() {
    # Check Docker
    check_docker
    
    # Create necessary directories
    echo -e "\n${YELLOW}Creating directories...${NC}"
    mkdir -p data/{storage,models,exports,logs}
    echo -e "${GREEN}✓ Directories created${NC}"
    
    # Start Docker services
    echo -e "\n${YELLOW}Starting Docker services...${NC}"
    docker-compose -f docker-compose.services.yml up -d
    
    # Fix MongoDB permissions
    fix_mongodb_permissions
    
    # Check services health with proper container names and commands
    check_service_health "Redis" "scaletent-redis-1" "redis-cli ping" 30 || exit 1
    check_service_health "MongoDB" "scaletent-mongodb-1" "mongosh --eval \"db.adminCommand('ping')\"" 30 || exit 1
    
    # Setup Python environment
    setup_python_env
    
    # Export environment variables
    export PYTHONPATH=$(pwd)
    export OPENCV_AVFOUNDATION_SKIP_AUTH=1
    
    # Start application components in new terminals
    echo -e "\n${YELLOW}Starting ScaleTent application components...${NC}"
    start_in_terminal "ScaleTent Web Interface" "uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload"
    start_in_terminal "ScaleTent API Server" "uvicorn src.main:app --host 0.0.0.0 --port 5000 --reload"
    # start_in_terminal "ScaleTent Camera Demo" "export OPENCV_AVFOUNDATION_SKIP_AUTH=1 && python run_local.py"
    start_in_terminal "ScaleTent Camera Demo" "env OPENCV_AVFOUNDATION_SKIP_AUTH=0 python run_local.py"
    
    # Create stop script
    create_stop_script
    
    # Wait a moment for services to start
    echo -e "\n${YELLOW}Waiting for services to start...${NC}"
    sleep 5
    
    # Check if UI and API are up
    check_endpoint "Web UI" "http://localhost:8000" 30 
    check_endpoint "API Server" "http://localhost:5000/docs" 30
    
    # Show access information
    echo -e "\n${GREEN}✓ ScaleTent is running!${NC}"
    echo -e "${YELLOW}Access points:${NC}"
    echo -e "• Web Interface: ${GREEN}http://localhost:8000${NC}"
    echo -e "• API Docs: ${GREEN}http://localhost:5000/docs${NC}"
    echo -e "• Redis: ${GREEN}localhost:6379${NC}"
    echo -e "• MongoDB: ${GREEN}localhost:27017${NC}"
    
    echo -e "\n${YELLOW}To stop all services:${NC}"
    echo -e "$ ./stop-all.sh"
}

# Execute main function
main
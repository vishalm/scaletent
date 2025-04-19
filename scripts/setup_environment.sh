#!/bin/bash
# ScaleTent Setup Environment Script
# This script installs all necessary dependencies and sets up the environment for the ScaleTent system

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== ScaleTent Environment Setup =====${NC}"
echo -e "${BLUE}This script will set up the environment for the ScaleTent system.${NC}"
echo ""

# Check if run as root
if [ "$EUID" -eq 0 ]; then
  echo -e "${YELLOW}Warning: Running as root is not recommended. Consider using a non-root user.${NC}"
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Setup aborted.${NC}"
    exit 1
  fi
fi

# Check operating system
echo -e "${BLUE}Checking system...${NC}"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  OS="linux"
  echo -e "${GREEN}Linux detected.${NC}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  OS="macos"
  echo -e "${GREEN}macOS detected.${NC}"
else
  echo -e "${RED}Unsupported operating system: $OSTYPE${NC}"
  echo -e "${YELLOW}This script is designed for Linux and macOS. For Windows, please follow the manual setup instructions.${NC}"
  exit 1
fi

# Check for Python 3.8+
echo -e "${BLUE}Checking Python version...${NC}"
if command -v python3 &>/dev/null; then
  PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
  PYTHON_CMD="python"
else
  echo -e "${RED}Python not found. Please install Python 3.8 or higher.${NC}"
  exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version | cut -d " " -f 2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
  echo -e "${RED}Python 3.8+ is required. Found Python $PYTHON_VERSION${NC}"
  exit 1
else
  echo -e "${GREEN}Python $PYTHON_VERSION detected.${NC}"
fi

# Check for pip
echo -e "${BLUE}Checking pip...${NC}"
if command -v pip3 &>/dev/null; then
  PIP_CMD="pip3"
elif command -v pip &>/dev/null; then
  PIP_CMD="pip"
else
  echo -e "${RED}pip not found. Please install pip.${NC}"
  exit 1
fi
echo -e "${GREEN}pip detected.${NC}"

# Check for virtual environment
echo -e "${BLUE}Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
  echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
  $PYTHON_CMD -m venv venv
  echo -e "${GREEN}Virtual environment created.${NC}"
else
  echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}Virtual environment activated.${NC}"

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
$PIP_CMD install --upgrade pip
echo -e "${GREEN}pip upgraded.${NC}"

# Install dependencies based on OS
echo -e "${BLUE}Installing system dependencies...${NC}"
if [ "$OS" == "linux" ]; then
  # Check for package manager
  if command -v apt-get &>/dev/null; then
    echo -e "${YELLOW}Using apt-get to install dependencies...${NC}"
    sudo apt-get update
    sudo apt-get install -y git cmake build-essential pkg-config libopencv-dev libavcodec-dev libavformat-dev libswscale-dev
    sudo apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
    
    # Check for CUDA
    if command -v nvcc &>/dev/null; then
      echo -e "${GREEN}CUDA detected.${NC}"
      echo -e "${YELLOW}Installing CUDA dependencies...${NC}"
      sudo apt-get install -y nvidia-cuda-toolkit
    else
      echo -e "${YELLOW}CUDA not detected. GPU acceleration will not be available.${NC}"
    fi
  elif command -v yum &>/dev/null; then
    echo -e "${YELLOW}Using yum to install dependencies...${NC}"
    sudo yum update -y
    sudo yum install -y git cmake gcc gcc-c++ opencv opencv-devel
    
    # Check for CUDA
    if command -v nvcc &>/dev/null; then
      echo -e "${GREEN}CUDA detected.${NC}"
      echo -e "${YELLOW}Please install CUDA dependencies manually using your package manager.${NC}"
    else
      echo -e "${YELLOW}CUDA not detected. GPU acceleration will not be available.${NC}"
    fi
  else
    echo -e "${YELLOW}Could not detect package manager. Please install the following dependencies manually:${NC}"
    echo "- git"
    echo "- cmake"
    echo "- build-essential"
    echo "- opencv"
    echo "- ffmpeg"
  fi
  
elif [ "$OS" == "macos" ]; then
  if command -v brew &>/dev/null; then
    echo -e "${YELLOW}Using Homebrew to install dependencies...${NC}"
    brew update
    brew install git cmake opencv ffmpeg
  else
    echo -e "${YELLOW}Homebrew not detected. Please install Homebrew or manually install the following dependencies:${NC}"
    echo "- git"
    echo "- cmake"
    echo "- opencv"
    echo "- ffmpeg"
  fi
fi

# Install Python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
$PIP_CMD install -r requirements.txt
echo -e "${GREEN}Python dependencies installed.${NC}"

# Download YOLOv8 model
echo -e "${BLUE}Downloading YOLOv8 model...${NC}"
mkdir -p data/models
if [ ! -f "data/models/yolov8n.pt" ]; then
  echo -e "${YELLOW}Downloading YOLOv8n model...${NC}"
  $PYTHON_CMD -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
  cp yolov8n.pt data/models/
  echo -e "${GREEN}YOLOv8 model downloaded.${NC}"
else
  echo -e "${GREEN}YOLOv8 model already exists.${NC}"
fi

# Create necessary directories
echo -e "${BLUE}Creating directory structure...${NC}"
mkdir -p data/profiles
mkdir -p data/detections
mkdir -p logs
mkdir -p config
echo -e "${GREEN}Directory structure created.${NC}"

# Copy example config if needed
if [ ! -f "config/config.yaml" ]; then
  echo -e "${BLUE}Copying example configuration...${NC}"
  cp config/config.example.yaml config/config.yaml
  echo -e "${GREEN}Example configuration copied to config/config.yaml${NC}"
  echo -e "${YELLOW}Please edit config/config.yaml to match your environment.${NC}"
else
  echo -e "${GREEN}Configuration file already exists.${NC}"
fi

# Set up privacy settings
if [ ! -f "config/privacy_settings.yaml" ]; then
  echo -e "${BLUE}Copying privacy settings...${NC}"
  cp config/privacy_settings.example.yaml config/privacy_settings.yaml
  echo -e "${GREEN}Privacy settings copied to config/privacy_settings.yaml${NC}"
  echo -e "${YELLOW}Please review config/privacy_settings.yaml for privacy controls.${NC}"
else
  echo -e "${GREEN}Privacy settings file already exists.${NC}"
fi

# Test the installation
echo -e "${BLUE}Testing installation...${NC}"
$PYTHON_CMD -c "
import cv2
import numpy as np
import torch
from ultralytics import YOLO
print('OpenCV version:', cv2.__version__)
print('NumPy version:', np.__version__)
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU device:', torch.cuda.get_device_name(0))
"

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${BLUE}===== ScaleTent Environment Setup Finished =====${NC}"
echo ""
echo -e "${YELLOW}To start the ScaleTent system:${NC}"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "2. Run the main application:"
echo "   python src/main.py --config config/config.yaml"
echo ""
echo -e "${YELLOW}For more information, refer to the documentation.${NC}"
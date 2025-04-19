#!/usr/bin/env python3
"""
Local demo script for ScaleTent using Mac's built-in camera
"""

import cv2
import yaml
import time
from pathlib import Path
import uvicorn

from src.recognition.face_detector import FaceDetector
from src.core.logger import setup_logger
from src.core.config import Config
from src.main import create_app

logger = setup_logger(__name__)

def main():
    # Initialize configuration
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    # Initialize configuration
    config = Config(str(config_path))
    
    # Create FastAPI application
    app = create_app(config_path=str(config_path))
    
    # Run the application
    uvicorn.run(
        app,
        host=config.get('api', 'rest.host', '0.0.0.0'),
        port=config.get('api', 'rest.port', 5000),
        log_level="info"
    )

if __name__ == "__main__":
    main() 
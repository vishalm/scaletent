#!/usr/bin/env python3
"""
ScaleTent - Real-time Meet & Greet System with YOLOv8 and OpenCV
Main application entry point
"""

import argparse
import asyncio
import os
import signal
import sys
import yaml
import logging
from pathlib import Path
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import Config
from src.core.logger import setup_logger
from src.core.device import get_device_from_config
from src.detection.detector import YOLODetector
from src.recognition.face_detector import FaceDetector
from src.recognition.matcher import FaceMatcher
from src.streaming.publisher import StreamPublisher
from src.core.camera import CameraManager
from src.api.websocket import WebSocketServer
from src.api.routes import router as api_router
from src.utils.ssl_utils import configure_ssl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ScaleTent Meet & Greet System')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    # Store components for cleanup
    app.state.components = {}
    
    try:
        # Load configuration
        config_file = Path(app.state.config_path)
        if not config_file.exists():
            logger.error(f"Configuration file not found: {app.state.config_path}")
            yield
            return

        # Initialize config with path - it will load the YAML internally
        config = Config(config_path=str(config_file))
        app.state.config = config
        
        if app.state.debug:
            config.set('general', 'debug', True)
        
        logger.info(f"Starting ScaleTent with configuration from {app.state.config_path}")
        
        # Get device configuration - default to 'auto' if not specified
        device_type = config.get('system', 'device', 'auto')
        device = get_device_from_config(device_type)
        logger.info(f"Using device: {device}")
        
        # Initialize detection components
        logger.info("Initializing YOLOv8 detector...")
        detector = YOLODetector(
            model_path=config.get('detection', 'model_path'),
            confidence_threshold=config.get('detection', 'confidence_threshold', 0.5),
            device=str(device),
            classes=config.get('detection', 'classes', [0])  # Default to person class
        )
        app.state.components['detector'] = detector
        
        # Initialize recognition components
        logger.info("Initializing face recognition components...")
        face_detector = FaceDetector(
            model_path=config.get('recognition', 'face_detector_model'),
            confidence_threshold=config.get('recognition', 'face_detection_threshold', 0.5)
        )
        
        # Initialize face matcher with device configuration
        face_matcher = FaceMatcher(
            embedder_model_path=config.get('recognition', 'embedder_model'),
            database_path=config.get('recognition', 'database_path'),
            similarity_threshold=config.get('recognition', 'similarity_threshold', 0.7),
            device=str(device)
        )
        
        app.state.components['face_detector'] = face_detector
        app.state.components['face_matcher'] = face_matcher
        
        # Initialize streaming
        logger.info("Initializing stream publisher...")
        publisher = StreamPublisher()
        app.state.components['publisher'] = publisher
        
        # Initialize camera manager
        logger.info("Initializing camera manager...")
        camera_manager = CameraManager(
            config=config,
            detector=detector,
            face_detector=face_detector,
            face_matcher=face_matcher,
            publisher=publisher
        )
        app.state.components['camera_manager'] = camera_manager
        
        # Initialize WebSocket server
        logger.info("Initializing WebSocket server...")
        websocket_server = WebSocketServer(
            host=config.get('api.websocket.host', '0.0.0.0'),
            port=config.get('api.websocket.port', 8765),
            publisher=publisher
        )
        app.state.components['websocket_server'] = websocket_server
        
        # Start components
        logger.info("Starting camera manager...")
        await camera_manager.start()
        
        logger.info("Starting WebSocket server...")
        await websocket_server.start()
        
        yield
        
        # Cleanup on shutdown
        logger.info("Shutting down components...")
        if 'camera_manager' in app.state.components:
            await app.state.components['camera_manager'].stop()
        
        if 'detector' in app.state.components:
            app.state.components['detector'].stop()
        
        if 'websocket_server' in app.state.components:
            await app.state.components['websocket_server'].stop()
        
    except Exception as e:
        logger.error(f"Error in lifespan: {e}", exc_info=True)
        yield
        return

# Create FastAPI application
app = FastAPI(
    title="ScaleTent API",
    description="API for ScaleTent Meet & Greet System",
    version="1.0.0",
    lifespan=lifespan
)

# Set default config path
app.state.config_path = "config/config.yaml"
app.state.debug = False

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Create required directories
static_dir = Path("src/web/static")
templates_dir = Path("src/web/templates")
static_dir.mkdir(parents=True, exist_ok=True)
templates_dir.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize templates
templates = Jinja2Templates(directory=str(templates_dir))

# Web interface routes
from fastapi import Request

@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
async def root(request: Request):
    """Root endpoint that serves the web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

# Include API routes
app.include_router(api_router)

if __name__ == "__main__":
    args = parse_arguments()
    
    # Update config path and debug mode from arguments
    app.state.config_path = args.config
    app.state.debug = args.debug
    
    try:
        # Configure SSL if needed
        if not configure_ssl():
            logger.warning("Failed to configure SSL certificates")
        
        # Run the application
        uvicorn.run(
            app,
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", 5000)),
            reload=args.debug,
            log_level="debug" if args.debug else "info"
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
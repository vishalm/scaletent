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
from pathlib import Path
import logging

from core.config import Config
from core.logger import setup_logger
from detection.detector import YOLODetector
from recognition.face_detector import FaceDetector
from recognition.matcher import FaceMatcher
from streaming.publisher import StreamPublisher
from core.camera import CameraManager
from api.websocket import WebSocketServer
from web.app import create_app
from utils.ssl_utils import configure_ssl

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

async def shutdown(signal, loop, app_components):
    """Cleanup function called on shutdown"""
    logger.info(f"Received exit signal {signal.name}...")
    
    # Stop components in reverse order
    if 'camera_manager' in app_components:
        logger.info("Stopping camera manager...")
        await app_components['camera_manager'].stop()
    
    if 'detector' in app_components:
        logger.info("Stopping detector...")
        app_components['detector'].stop()
    
    if 'websocket_server' in app_components:
        logger.info("Stopping WebSocket server...")
        await app_components['websocket_server'].stop()
    
    if 'web_app' in app_components:
        logger.info("Stopping web application...")
        await app_components['web_app'].shutdown()
    
    # Cancel tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()
    logger.info("Shutdown complete.")

async def run_application(config_path: str, debug: bool = False) -> int:
    """Main application runner"""
    # Configure SSL
    if not configure_ssl():
        logger.error("Failed to configure SSL certificates. Application may experience network issues.")
        return 1

    # Load configuration
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1

        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config = Config(**config_data)
        logger.info("Configuration loaded successfully")
        
        if debug:
            config.set_debug_mode(True)
        
        logger.info(f"Starting ScaleTent with configuration from {config_path}")
        
        # Store components for cleanup
        app_components = {}
        
        try:
            # Initialize detection components
            logger.info("Initializing YOLOv8 detector...")
            detector = YOLODetector(
                model_path=config.get('detection.model_path'),
                confidence_threshold=config.get('detection.confidence_threshold', 0.5),
                device=config.get('detection.device', 'cuda:0')
            )
            app_components['detector'] = detector
            
            # Initialize recognition components
            logger.info("Initializing face recognition components...")
            face_detector = FaceDetector(
                model_path=config.get('recognition.face_detector_model'),
                confidence_threshold=config.get('recognition.face_detection_threshold', 0.5)
            )
            
            face_matcher = FaceMatcher(
                embedder_model_path=config.get('recognition.embedder_model'),
                database_path=config.get('recognition.database_path'),
                similarity_threshold=config.get('recognition.similarity_threshold', 0.7)
            )
            
            app_components['face_detector'] = face_detector
            app_components['face_matcher'] = face_matcher
            
            # Initialize streaming
            logger.info("Initializing stream publisher...")
            publisher = StreamPublisher()
            app_components['publisher'] = publisher
            
            # Initialize camera manager
            logger.info("Initializing camera manager...")
            camera_manager = CameraManager(
                config=config,
                detector=detector,
                face_detector=face_detector,
                face_matcher=face_matcher,
                publisher=publisher
            )
            app_components['camera_manager'] = camera_manager
            
            # Initialize WebSocket server
            logger.info("Initializing WebSocket server...")
            websocket_server = WebSocketServer(
                host=config.get('api.websocket.host', '0.0.0.0'),
                port=config.get('api.websocket.port', 8765),
                publisher=publisher
            )
            app_components['websocket_server'] = websocket_server
            
            # Initialize web application
            logger.info("Initializing web application...")
            web_app = create_app(config)
            app_components['web_app'] = web_app
            
            # Start components
            logger.info("Starting camera manager...")
            await camera_manager.start()
            
            logger.info("Starting WebSocket server...")
            await websocket_server.start()
            
            logger.info("Starting web application...")
            await web_app.start()
            
            logger.info("ScaleTent system is running. Press Ctrl+C to exit.")
            
            # Set up signal handlers for graceful shutdown
            loop = asyncio.get_running_loop()
            signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
            for s in signals:
                loop.add_signal_handler(
                    s, lambda s=s: asyncio.create_task(shutdown(s, loop, app_components))
                )
            
            # Keep the main task running
            while True:
                await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in main application: {e}", exc_info=True)
            # Attempt shutdown of components
            loop = asyncio.get_running_loop()
            await shutdown(signal.SIGTERM, loop, app_components)
            return 1

        return 0
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        return 1

def main():
    """Main entry point of the application."""
    # Configure SSL certificates
    if not configure_ssl():
        logger.error("Failed to configure SSL certificates. Application may experience network issues.")
    
    # Rest of your application initialization code here
    logger.info("Starting Scaletent application...")

if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        exit_code = asyncio.run(run_application(args.config, args.debug))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Exiting...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
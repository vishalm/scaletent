#!/usr/bin/env python3
"""
Local demo script for ScaleTent using Mac's built-in camera
"""

import cv2
import yaml
import time
from pathlib import Path

from src.recognition.face_detector import FaceDetector
from src.core.logger import setup_logger

logger = setup_logger(__name__)

def main():
    # Load configuration
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize face detector with MediaPipe backend
    detector = FaceDetector(
        backend="mediapipe",
        confidence_threshold=config["recognition"]["face_detection_threshold"],
        device=config["system"]["device"]["type"]
    )
    
    # Initialize camera
    camera_config = config["cameras"][0]  # Using first camera (Mac built-in)
    cap = cv2.VideoCapture(0)  # 0 is usually the built-in camera
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["height"])
    cap.set(cv2.CAP_PROP_FPS, camera_config["fps"])
    
    logger.info("Starting camera feed...")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break
            
            # Detect faces
            faces = detector.detect(frame)
            
            # Draw detections
            frame_with_detections = detector.draw_face_detections(frame, faces)
            
            # Show frame
            cv2.imshow("ScaleTent - Face Detection", frame_with_detections)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Control frame rate
            time.sleep(1/camera_config["processing_fps"])
    
    except KeyboardInterrupt:
        logger.info("Stopping camera feed...")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
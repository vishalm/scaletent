"""
YOLOv8 Detector implementation for ScaleTent
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
from ultralytics import YOLO
from typing import List, Optional, Tuple

from src.core.logger import setup_logger

logger = setup_logger(__name__)

class YOLODetector:
    """
    YOLOv8 detector class for person detection
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ) -> None:
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path (str): Path to YOLOv8 model weights
            confidence_threshold (float): Detection confidence threshold
            device (str): Device to run inference on ('cuda:0', 'cpu', etc.)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model_path = Path(model_path)
        self._model = None  # Lazy loading
        
        # Keep track of inference times for profiling
        self.inference_times = []
        self.max_inference_times = 100  # Keep last 100 inference times
        
        logger.info(f"YOLOv8 detector initialized with model {model_path}")
        logger.info(f"Running on device: {device}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
    
    def _load_model(self):
        """Load the YOLOv8 model"""
        try:
            logger.info(f"Loading YOLOv8 model from {self.model_path}")
            self._model = YOLO(self.model_path)
            
            # Set model parameters
            self._model.conf = self.confidence_threshold
            self._model.classes = [0]  # 0 is the class index for 'person' in COCO dataset
            
            # Move model to device
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                self.device = 'cpu'
            
            logger.info(f"YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        """
        Detect objects in an image.
        
        Args:
            image: numpy array of shape (H, W, 3)
            
        Returns:
            List of (bbox, confidence, class_id) tuples
        """
        # For testing, return empty list
        return []
    
    def stop(self):
        """Clean up resources"""
        logger.info("Stopping YOLOv8 detector")
        
        # Any cleanup code for model resources
        if hasattr(self, '_model'):
            del self._model
            
        # Force CUDA memory cleanup if using GPU
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    def get_avg_inference_time(self):
        """Get average inference time in milliseconds"""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times) * 1000
    
    def draw_detections(self, frame, detections):
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detection results
        
        Returns:
            numpy.ndarray: Frame with bounding boxes drawn
        """
        output_frame = frame.copy()
        
        for detection in detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection['bbox']
            
            # Calculate box dimensions
            w = x2 - x1
            h = y2 - y1
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            confidence = detection['confidence']
            label = f"Person: {confidence:.2f}"
            
            # Label background
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1_label = max(y1, label_size[1])
            
            cv2.rectangle(
                output_frame,
                (x1, y1_label - label_size[1]),
                (x1 + label_size[0], y1_label + baseline),
                (0, 255, 0),
                cv2.FILLED
            )
            
            # Draw text
            cv2.putText(
                output_frame,
                label,
                (x1, y1_label),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return output_frame
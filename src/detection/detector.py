"""
YOLOv8 Detector implementation for ScaleTent
"""

import cv2
import numpy as np
import torch
import time
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
import logging

from src.core.logger import setup_logger

logger = logging.getLogger(__name__)

class YOLODetector:
    """
    YOLOv8 detector class for person detection
    """
    
    SUPPORTED_DEVICES = {
        'cpu': 'CPU',
        'cuda': 'NVIDIA GPU with CUDA',
        'mps': 'Apple Silicon GPU',
    }
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: str = 'cpu',
        classes: Optional[List[int]] = None
    ) -> None:
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path (str): Path to YOLOv8 model weights
            confidence_threshold (float): Detection confidence threshold
            device (str): Device to run inference on ('cpu', 'cuda', 'mps')
            classes: List of class IDs to detect (None for all classes)
        """
        self.confidence_threshold = confidence_threshold
        self.device = self._validate_device(device)
        self.model_path = Path(model_path)
        self.classes = classes if classes is not None else [0]  # Default to person class
        self.model = None
        self.is_running = False
        
        # Keep track of inference times for profiling
        self.inference_times = []
        self.max_inference_times = 100  # Keep last 100 inference times
        
        logger.info(f"YOLOv8 detector initialized with model {model_path}")
        logger.info(f"Running on device: {device} ({self.SUPPORTED_DEVICES[device]})")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
        self._load_model()
    
    def _validate_device(self, device: str) -> str:
        """Validate and return appropriate device for inference.
        
        Args:
            device: Requested device ('cpu', 'cuda', 'mps')
            
        Returns:
            Validated device string
        """
        device = device.lower()
        
        if device not in self.SUPPORTED_DEVICES:
            logger.warning(f"Unsupported device '{device}'. Falling back to CPU.")
            return 'cpu'
            
        # Check device availability
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return 'cpu'
        elif device == 'mps' and not getattr(torch.backends, 'mps', None):
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return 'cpu'
            
        logger.info(f"Using device: {device} ({self.SUPPORTED_DEVICES[device]})")
        return device
    
    def _load_model(self):
        """Load the YOLOv8 model"""
        try:
            logger.info(f"Loading YOLOv8 model from {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Set model parameters
            self.model.conf = self.confidence_threshold
            self.model.classes = self.classes
            
            logger.info("Model loaded successfully")
            self.is_running = True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Perform detection on a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: Detection confidence
                - class_id: Class ID
        """
        if not self.is_running or self.model is None:
            logger.warning("Detector is not running")
            return []
            
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                classes=self.classes,
                device=self.device
            )
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        'bbox': box.xyxy[0].tolist(),
                        'confidence': float(box.conf),
                        'class_id': int(box.cls)
                    }
                    detections.append(detection)
                    
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return []
    
    def stop(self):
        """Clean up resources"""
        logger.info("Stopping YOLOv8 detector")
        
        self.is_running = False
        if self.model is not None:
            self.model.cpu()  # Move model to CPU to free GPU memory
            self.model = None
            
        # Force CUDA memory cleanup if using GPU
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    def get_avg_inference_time(self):
        """Get average inference time in milliseconds"""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times) * 1000
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw detection boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections from detect()
            color: BGR color tuple for boxes
            thickness: Line thickness
            
        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence
            label = f"{conf:.2f}"
            cv2.putText(
                output_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness
            )
        
        return output_frame
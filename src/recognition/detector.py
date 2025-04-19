"""
Object detection module using YOLOv8
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO

from src.core.logger import setup_logger

logger = setup_logger(__name__)

class Detector:
    """
    Object detector class using YOLOv8 for person detection.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        classes: Optional[List[int]] = None
    ):
        """
        Initialize the detector.
        
        Args:
            model_path (str): Path to YOLOv8 model file
            confidence_threshold (float): Detection confidence threshold
            device (str): Device to run inference on ('cpu', 'cuda:0', etc.)
            classes (Optional[List[int]]): List of class IDs to detect (None for all)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.classes = classes if classes is not None else [0]  # Default to person class only
        
        try:
            # Load YOLOv8 model
            self.model = YOLO(model_path)
            self.model.to(device)
            logger.info(f"Loaded YOLOv8 model from {model_path} on {device}")
            
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame.
        
        Args:
            frame (np.ndarray): Input frame (BGR format)
            
        Returns:
            List[Dict[str, Any]]: List of detections, each containing:
                - bbox: [x1, y1, x2, y2] (normalized coordinates)
                - confidence: Detection confidence
                - class_id: Class ID
                - class_name: Class name
        """
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                classes=self.classes,
                verbose=False
            )
            
            detections = []
            
            # Process results
            if len(results) > 0:
                result = results[0]  # Get first image result
                
                # Convert detections to normalized coordinates
                height, width = frame.shape[:2]
                
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxn[0].cpu().numpy()  # Normalized coordinates
                    
                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection boxes and labels on frame.
        
        Args:
            frame (np.ndarray): Input frame
            detections (List[Dict[str, Any]]): List of detections
            color (Tuple[int, int, int]): Box color in BGR format
            thickness (int): Line thickness
            
        Returns:
            np.ndarray: Frame with drawn detections
        """
        height, width = frame.shape[:2]
        output = frame.copy()
        
        for det in detections:
            # Get box coordinates
            x1, y1, x2, y2 = det["bbox"]
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1 = max(y1, label_size[1])
            
            cv2.rectangle(
                output,
                (x1, y1 - label_size[1]),
                (x1 + label_size[0], y1 + baseline),
                color,
                cv2.FILLED
            )
            
            cv2.putText(
                output,
                label,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return output

    @property
    def available_classes(self) -> Dict[int, str]:
        """Get available class IDs and names."""
        return self.model.names 
"""
Face detection module for ScaleTent
"""

import cv2
import numpy as np
import time
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any

from src.core.logger import setup_logger

logger = setup_logger(__name__)

class FaceDetector:
    """
    Face detector implementation for ScaleTent
    
    This class uses MediaPipe Face Detection by default, with options for:
    1. MediaPipe Face Detection (default)
    2. RetinaFace (optional)
    """
    
    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        backend: str = "mediapipe",
        device: str = "cpu"
    ) -> None:
        """
        Initialize face detector
        
        Args:
            model_path (str, optional): Path to face detection model (only needed for some backends)
            confidence_threshold (float): Detection confidence threshold
            backend (str): Detection backend ('mediapipe' or 'retinaface')
            device (str): Device to run inference on ('cuda:0', 'cpu', etc.)
        """
        self.confidence_threshold = confidence_threshold
        self.backend = backend.lower()
        self.device = device
        
        if self.backend not in ["mediapipe", "retinaface"]:
            logger.warning(f"Unsupported backend '{backend}', falling back to MediaPipe")
            self.backend = "mediapipe"
        
        # Initialize model based on backend
        self._load_model()
        
        # Keep track of inference times
        self.inference_times = []
        self.max_inference_times = 100
        
        logger.info(f"Face detector initialized with backend: {self.backend}")
        logger.info(f"Running on device: {device}")
    
    def _load_model(self):
        """Load the appropriate face detection model"""
        try:
            if self.backend == "mediapipe":
                logger.info("Loading MediaPipe face detector")
                try:
                    import mediapipe as mp
                    self.mp_face_detection = mp.solutions.face_detection
                    self.model = self.mp_face_detection.FaceDetection(
                        model_selection=1,  # 0 for close range, 1 for far range
                        min_detection_confidence=self.confidence_threshold
                    )
                    logger.info("MediaPipe face detector loaded successfully")
                except ImportError:
                    logger.error("MediaPipe not installed. Please install with: pip install mediapipe")
                    raise
            
            elif self.backend == "retinaface":
                logger.info("Loading RetinaFace detector")
                try:
                    from retinaface import RetinaFace
                    self.model = RetinaFace(
                        gpu_id=0 if self.device.startswith("cuda") else -1
                    )
                    logger.info("RetinaFace detector loaded successfully")
                except ImportError:
                    logger.error("RetinaFace not installed. Please install with: pip install retina-face")
                    raise
        
        except Exception as e:
            logger.error(f"Failed to load face detection model: {e}")
            raise
    
    def detect_faces(self, frame: np.ndarray, person_detections: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect faces in a frame
        
        Args:
            frame (numpy.ndarray): Input frame
            person_detections (list, optional): Person detections to limit face detection regions
        
        Returns:
            list: List of face detection results with format
                 [{'bbox': [x1, y1, x2, y2], 'confidence': conf, 'id': 'face-X'}]
        """
        if frame is None:
            logger.warning("Received empty frame for face detection")
            return []
        
        start_time = time.time()
        
        try:
            face_detections = []
            
            # If person detections are provided, only search for faces within those regions
            if person_detections:
                for person in person_detections:
                    # Get person bounding box with integer coordinates
                    x1, y1, x2, y2 = map(int, person['bbox'])
                    
                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    # Extract person region
                    person_region = frame[y1:y2, x1:x2]
                    if person_region.size == 0:
                        continue
                    
                    # Detect faces in person region
                    region_faces = self._detect_faces_in_frame(person_region)
                    
                    # Adjust face coordinates to original frame
                    for face in region_faces:
                        face_x1, face_y1, face_x2, face_y2 = face['bbox']
                        face['bbox'] = [
                            int(x1 + face_x1),
                            int(y1 + face_y1),
                            int(x1 + face_x2),
                            int(y1 + face_y2)
                        ]
                        face['person_id'] = person.get('id')
                        face_detections.append(face)
            else:
                # Detect faces in entire frame
                face_detections = self._detect_faces_in_frame(frame)
            
            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > self.max_inference_times:
                self.inference_times = self.inference_times[-self.max_inference_times:]
            
            logger.debug(f"Detected {len(face_detections)} faces in {inference_time:.4f} seconds")
            
            return face_detections
        
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return []
    
    def _detect_faces_in_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a frame using the selected backend
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            list: List of face detection results
        """
        if self.backend == "mediapipe":
            return self._detect_faces_mediapipe(frame)
        elif self.backend == "retinaface":
            return self._detect_faces_retinaface(frame)
        else:
            logger.error(f"Unsupported face detection backend: {self.backend}")
            return []
    
    def _detect_faces_mediapipe(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MediaPipe"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # Run detection
        results = self.model.process(rgb_frame)
        
        faces = []
        face_count = 0
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x1 = int(bbox.xmin * width)
                y1 = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                x2 = x1 + w
                y2 = y1 + h
                
                # Ensure box is within frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Skip invalid detections
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Add face detection
                face_count += 1
                
                faces.append({
                    'id': f"face-{face_count}",
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(detection.score[0])
                })
        
        return faces
    
    def _detect_faces_retinaface(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using RetinaFace"""
        # RetinaFace expects BGR format, which is the OpenCV default
        height, width = frame.shape[:2]
        
        # Run detection
        faces_dict = self.model.detect_faces(frame)
        
        faces = []
        face_count = 0
        
        # Process detections
        if faces_dict:
            for face_key in faces_dict:
                face_data = faces_dict[face_key]
                
                # Get bounding box and confidence
                x1, y1, x2, y2 = map(int, face_data['facial_area'])
                confidence = face_data['score']
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Ensure box is within frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Skip invalid detections
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Add face detection
                face_count += 1
                
                faces.append({
                    'id': f"face-{face_count}",
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'landmarks': face_data.get('landmarks', None)
                })
        
        return faces
    
    def get_avg_inference_time(self) -> float:
        """Get average inference time in milliseconds"""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times) * 1000
    
    def draw_face_detections(self, frame: np.ndarray, face_detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw face detection bounding boxes on frame
        
        Args:
            frame (numpy.ndarray): Input frame
            face_detections (list): List of face detection results
        
        Returns:
            numpy.ndarray: Frame with bounding boxes drawn
        """
        output_frame = frame.copy()
        
        for detection in face_detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection['bbox']
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            confidence = detection['confidence']
            label = f"Face: {confidence:.2f}"
            
            # Label background
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1_label = max(y1, label_size[1])
            
            cv2.rectangle(
                output_frame,
                (x1, y1_label - label_size[1]),
                (x1 + label_size[0], y1_label + baseline),
                (0, 0, 255),
                cv2.FILLED
            )
            
            # Draw text
            cv2.putText(
                output_frame,
                label,
                (x1, y1_label),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Draw landmarks if available (RetinaFace)
            if 'landmarks' in detection and detection['landmarks'] is not None:
                landmarks = detection['landmarks']
                for point in landmarks.values():
                    x, y = point
                    cv2.circle(output_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        return output_frame

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Main detection method that should be used by external callers.
        Detects faces in the given frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of face detections with format
                 [{'bbox': [x1, y1, x2, y2], 'confidence': conf, 'id': 'face-X'}]
        """
        return self.detect_faces(frame)
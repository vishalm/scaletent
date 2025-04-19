"""
Face detection module for ScaleTent
"""

import cv2
import numpy as np
import time
import torch
from pathlib import Path
from typing import List, Tuple

from src.core.logger import setup_logger

logger = setup_logger(__name__)

class FaceDetector:
    """
    Face detector implementation for ScaleTent
    
    This class can use different face detection backends:
    1. OpenCV DNN Face Detector (SSD)
    2. MediaPipe Face Detection
    3. RetinaFace
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, backend="opencv", device="cuda:0") -> None:
        """
        Initialize face detector
        
        Args:
            model_path (str): Path to face detection model
            confidence_threshold (float): Detection confidence threshold
            backend (str): Detection backend ('opencv', 'mediapipe', or 'retinaface')
            device (str): Device to run inference on ('cuda:0', 'cpu', etc.)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.backend = backend
        self.device = device
        
        # Initialize model based on backend
        self._load_model()
        
        # Keep track of inference times
        self.inference_times = []
        self.max_inference_times = 100
        
        logger.info(f"Face detector initialized with backend: {backend}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Running on device: {device}")
    
    def _load_model(self):
        """Load the appropriate face detection model"""
        try:
            if self.backend == "opencv":
                logger.info("Loading OpenCV DNN face detector")
                self.model = cv2.dnn.readNetFromCaffe(
                    str(self.model_path.parent / "deploy.prototxt"),
                    str(self.model_path)
                )
                
                # Set computation backend
                if self.device.startswith("cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    logger.info("Using CUDA backend for OpenCV DNN")
                    self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                else:
                    logger.info("Using CPU backend for OpenCV DNN")
                    self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            elif self.backend == "mediapipe":
                logger.info("Loading MediaPipe face detector")
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.model = self.mp_face_detection.FaceDetection(
                    model_selection=1,  # 0 for close range, 1 for far range
                    min_detection_confidence=self.confidence_threshold
                )
            
            elif self.backend == "retinaface":
                logger.info("Loading RetinaFace detector")
                from retinaface import RetinaFace
                self.model = RetinaFace(
                    gpu_id=0 if self.device.startswith("cuda") else -1
                )
            
            else:
                raise ValueError(f"Unsupported face detection backend: {self.backend}")
                
            logger.info("Face detection model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load face detection model: {e}")
            raise
    
    def detect_faces(self, frame, person_detections=None):
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
                    # Extract person bounding box
                    x1, y1, x2, y2 = person['bbox']
                    
                    # Add some padding
                    height, width = frame.shape[:2]
                    x1 = max(0, x1 - 20)
                    y1 = max(0, y1 - 20)
                    x2 = min(width, x2 + 20)
                    y2 = min(height, y2 + 20)
                    
                    # Extract region of interest
                    person_roi = frame[y1:y2, x1:x2]
                    
                    # Skip if ROI is empty
                    if person_roi.size == 0:
                        continue
                    
                    # Detect faces in person ROI
                    roi_faces = self._detect_faces_in_frame(person_roi)
                    
                    # Adjust coordinates to full frame
                    for face in roi_faces:
                        face_x1, face_y1, face_x2, face_y2 = face['bbox']
                        face['bbox'] = [
                            x1 + face_x1,
                            y1 + face_y1,
                            x1 + face_x2,
                            y1 + face_y2
                        ]
                        
                        # Link to person detection
                        face['person_id'] = person['id']
                        
                        face_detections.append(face)
            
            else:
                # Detect faces in full frame
                face_detections = self._detect_faces_in_frame(frame)
            
            # Record inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only last N inference times
            if len(self.inference_times) > self.max_inference_times:
                self.inference_times = self.inference_times[-self.max_inference_times:]
            
            logger.debug(f"Detected {len(face_detections)} faces in {inference_time:.4f} seconds")
            
            return face_detections
        
        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return []
    
    def _detect_faces_in_frame(self, frame):
        """
        Detect faces in a frame using the selected backend
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            list: List of face detection results
        """
        if self.backend == "opencv":
            return self._detect_faces_opencv(frame)
        elif self.backend == "mediapipe":
            return self._detect_faces_mediapipe(frame)
        elif self.backend == "retinaface":
            return self._detect_faces_retinaface(frame)
        else:
            logger.error(f"Unsupported face detection backend: {self.backend}")
            return []
    
    def _detect_faces_opencv(self, frame):
        """Detect faces using OpenCV DNN"""
        height, width = frame.shape[:2]
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        # Set input and forward pass
        self.model.setInput(blob)
        detections = self.model.forward()
        
        faces = []
        face_count = 0
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < self.confidence_threshold:
                continue
            
            # Get bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype(int)
            
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
                'confidence': float(confidence)
            })
        
        return faces
    
    def _detect_faces_mediapipe(self, frame):
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
    
    def _detect_faces_retinaface(self, frame):
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
                x1, y1, x2, y2 = face_data['facial_area']
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
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'landmarks': face_data.get('landmarks', None)
                })
        
        return faces
    
    def get_avg_inference_time(self):
        """Get average inference time in milliseconds"""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times) * 1000
    
    def draw_face_detections(self, frame, face_detections):
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

    def detect(self, image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Detect faces in image."""
        return []
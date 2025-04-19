"""
Video preprocessing module for ScaleTent
Handles frame preprocessing to improve detection performance
"""

import cv2
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional

from core.logger import setup_logger

logger = setup_logger(__name__)

class VideoPreprocessor:
    """
    Video frame preprocessor for ScaleTent
    
    This class handles:
    1. Frame resizing
    2. Noise reduction
    3. Contrast enhancement
    4. Motion detection
    5. ROI (Region of Interest) extraction
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor
        
        Args:
            config (dict): Preprocessor configuration
        """
        self.config = config
        
        # Default configuration
        self.target_width = config.get("target_width", 640)
        self.target_height = config.get("target_height", 480)
        self.maintain_aspect_ratio = config.get("maintain_aspect_ratio", True)
        
        # Noise reduction settings
        self.enable_noise_reduction = config.get("enable_noise_reduction", True)
        self.noise_reduction_method = config.get("noise_reduction_method", "gaussian")
        self.gaussian_kernel_size = config.get("gaussian_kernel_size", (5, 5))
        self.gaussian_sigma = config.get("gaussian_sigma", 0)
        
        # Contrast enhancement settings
        self.enable_contrast_enhancement = config.get("enable_contrast_enhancement", False)
        self.contrast_method = config.get("contrast_method", "clahe")
        self.clahe_clip_limit = config.get("clahe_clip_limit", 2.0)
        self.clahe_tile_grid_size = config.get("clahe_tile_grid_size", (8, 8))
        
        # Motion detection settings
        self.enable_motion_detection = config.get("enable_motion_detection", False)
        self.motion_detection_method = config.get("motion_detection_method", "mog2")
        self.motion_history_length = config.get("motion_history_length", 5)
        self.motion_threshold = config.get("motion_threshold", 25)
        
        # ROI settings
        self.enable_roi = config.get("enable_roi", False)
        self.roi_areas = config.get("roi_areas", [])
        
        # Initialize preprocessor components
        self._init_preprocessor()
        
        # Stateful components
        self.previous_frames = []
        self.background_subtractor = None
        
        logger.info("Video preprocessor initialized")
    
    def _init_preprocessor(self):
        """Initialize preprocessor components"""
        # Initialize CLAHE for contrast enhancement
        if self.enable_contrast_enhancement and self.contrast_method == "clahe":
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size
            )
        
        # Initialize background subtractor for motion detection
        if self.enable_motion_detection:
            if self.motion_detection_method == "mog2":
                self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=self.motion_history_length * 30,  # Assuming 30 FPS
                    varThreshold=self.motion_threshold,
                    detectShadows=False
                )
            elif self.motion_detection_method == "knn":
                self.background_subtractor = cv2.createBackgroundSubtractorKNN(
                    history=self.motion_history_length * 30,  # Assuming 30 FPS
                    dist2Threshold=self.motion_threshold,
                    detectShadows=False
                )
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess a video frame
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            tuple: (processed_frame, metadata)
        """
        if frame is None:
            logger.warning("Received empty frame for preprocessing")
            return None, {}
        
        try:
            # Start timing
            start_time = time.time()
            
            # Working copy of the frame
            processed_frame = frame.copy()
            
            # Metadata to return
            metadata = {
                "original_width": frame.shape[1],
                "original_height": frame.shape[0],
                "processing_steps": []
            }
            
            # 1. Resize frame
            processed_frame, resize_meta = self._resize_frame(processed_frame)
            metadata["processing_steps"].append({"type": "resize", "params": resize_meta})
            
            # 2. Apply noise reduction
            if self.enable_noise_reduction:
                processed_frame, noise_meta = self._reduce_noise(processed_frame)
                metadata["processing_steps"].append({"type": "noise_reduction", "params": noise_meta})
            
            # 3. Apply contrast enhancement
            if self.enable_contrast_enhancement:
                processed_frame, contrast_meta = self._enhance_contrast(processed_frame)
                metadata["processing_steps"].append({"type": "contrast_enhancement", "params": contrast_meta})
            
            # 4. Apply motion detection
            motion_mask = None
            if self.enable_motion_detection:
                motion_mask, motion_meta = self._detect_motion(processed_frame)
                metadata["processing_steps"].append({"type": "motion_detection", "params": motion_meta})
            
            # 5. Apply ROI masking
            if self.enable_roi and self.roi_areas:
                processed_frame, roi_meta = self._apply_roi(processed_frame, motion_mask)
                metadata["processing_steps"].append({"type": "roi", "params": roi_meta})
            
            # Add final dimensions
            metadata["processed_width"] = processed_frame.shape[1]
            metadata["processed_height"] = processed_frame.shape[0]
            
            # Add processing time
            processing_time = time.time() - start_time
            metadata["processing_time"] = processing_time
            
            return processed_frame, metadata
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return frame, {"error": str(e)}
    
    def _resize_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resize frame to target dimensions
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            tuple: (resized_frame, metadata)
        """
        original_height, original_width = frame.shape[:2]
        target_width, target_height = self.target_width, self.target_height
        
        # Calculate new dimensions
        if self.maintain_aspect_ratio:
            # Calculate aspect ratio
            aspect_ratio = original_width / original_height
            
            # Determine new dimensions based on target
            if original_width > original_height:
                # Landscape orientation
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
                
                # Check if height exceeds target
                if new_height > target_height:
                    new_height = target_height
                    new_width = int(new_height * aspect_ratio)
            else:
                # Portrait or square orientation
                new_height = target_height
                new_width = int(new_height * aspect_ratio)
                
                # Check if width exceeds target
                if new_width > target_width:
                    new_width = target_width
                    new_height = int(new_width / aspect_ratio)
        else:
            # Force resize to target dimensions
            new_width, new_height = target_width, target_height
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Return metadata
        metadata = {
            "original_width": original_width,
            "original_height": original_height,
            "new_width": new_width,
            "new_height": new_height,
            "maintain_aspect_ratio": self.maintain_aspect_ratio
        }
        
        return resized_frame, metadata
    
    def _reduce_noise(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply noise reduction to frame
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            tuple: (processed_frame, metadata)
        """
        metadata = {
            "method": self.noise_reduction_method
        }
        
        if self.noise_reduction_method == "gaussian":
            processed_frame = cv2.GaussianBlur(
                frame,
                self.gaussian_kernel_size,
                self.gaussian_sigma
            )
            metadata["kernel_size"] = self.gaussian_kernel_size
            metadata["sigma"] = self.gaussian_sigma
            
        elif self.noise_reduction_method == "bilateral":
            processed_frame = cv2.bilateralFilter(
                frame,
                d=9,
                sigmaColor=75,
                sigmaSpace=75
            )
            metadata["d"] = 9
            metadata["sigma_color"] = 75
            metadata["sigma_space"] = 75
            
        elif self.noise_reduction_method == "median":
            processed_frame = cv2.medianBlur(frame, 5)
            metadata["kernel_size"] = 5
            
        else:
            # Default to no processing if method not recognized
            processed_frame = frame
            metadata["method"] = "none"
            metadata["reason"] = f"Method {self.noise_reduction_method} not recognized"
        
        return processed_frame, metadata
    
    def _enhance_contrast(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Enhance contrast in frame
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            tuple: (processed_frame, metadata)
        """
        metadata = {
            "method": self.contrast_method
        }
        
        if self.contrast_method == "clahe":
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l_enhanced = self.clahe.apply(l)
            
            # Merge channels back
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            
            # Convert back to BGR
            processed_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            metadata["clip_limit"] = self.clahe_clip_limit
            metadata["tile_grid_size"] = self.clahe_tile_grid_size
            
        elif self.contrast_method == "histogram_equalization":
            # Convert to YUV color space
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            
            # Apply histogram equalization to Y channel
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            
            # Convert back to BGR
            processed_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
        else:
            # Default to no processing if method not recognized
            processed_frame = frame
            metadata["method"] = "none"
            metadata["reason"] = f"Method {self.contrast_method} not recognized"
        
        return processed_frame, metadata
    
    def _detect_motion(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect motion in frame
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            tuple: (motion_mask, metadata)
        """
        metadata = {
            "method": self.motion_detection_method,
            "motion_detected": False
        }
        
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.motion_detection_method in ["mog2", "knn"]:
            # Apply background subtraction
            motion_mask = self.background_subtractor.apply(gray)
            
            # Apply threshold to remove shadows
            _, motion_mask = cv2.threshold(motion_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Count non-zero pixels to determine motion
            motion_pixels = cv2.countNonZero(motion_mask)
            motion_percentage = motion_pixels / (frame.shape[0] * frame.shape[1])
            
            # Determine if motion is detected
            metadata["motion_pixels"] = motion_pixels
            metadata["motion_percentage"] = motion_percentage
            metadata["motion_detected"] = motion_percentage > 0.01  # 1% of frame
            
        elif self.motion_detection_method == "frame_diff":
            # Store current frame for next comparison
            if not self.previous_frames:
                self.previous_frames.append(gray)
                motion_mask = np.zeros_like(gray)
            else:
                # Calculate absolute difference between current and previous frame
                frame_diff = cv2.absdiff(gray, self.previous_frames[0])
                
                # Apply threshold
                _, motion_mask = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
                
                # Update previous frame
                self.previous_frames[0] = gray
                
                # Count non-zero pixels to determine motion
                motion_pixels = cv2.countNonZero(motion_mask)
                motion_percentage = motion_pixels / (frame.shape[0] * frame.shape[1])
                
                # Determine if motion is detected
                metadata["motion_pixels"] = motion_pixels
                metadata["motion_percentage"] = motion_percentage
                metadata["motion_detected"] = motion_percentage > 0.01  # 1% of frame
        else:
            # Default to no motion detection if method not recognized
            motion_mask = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            metadata["method"] = "none"
            metadata["reason"] = f"Method {self.motion_detection_method} not recognized"
        
        return motion_mask, metadata
    
    def _apply_roi(self, frame: np.ndarray, motion_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Region of Interest (ROI) masking
        
        Args:
            frame (numpy.ndarray): Input frame
            motion_mask (numpy.ndarray, optional): Motion mask for dynamic ROI
        
        Returns:
            tuple: (processed_frame, metadata)
        """
        metadata = {
            "roi_count": len(self.roi_areas),
            "roi_applied": False
        }
        
        # If no ROIs defined, return original frame
        if not self.roi_areas:
            return frame, metadata
        
        # Create a mask for ROIs
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Draw ROIs on mask
        for roi in self.roi_areas:
            roi_type = roi.get("type", "rectangle")
            
            if roi_type == "rectangle":
                x1, y1, x2, y2 = roi["coordinates"]
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                
            elif roi_type == "polygon":
                points = np.array(roi["coordinates"], np.int32)
                cv2.fillPoly(mask, [points], 255)
                
            elif roi_type == "circle":
                center_x, center_y, radius = roi["coordinates"]
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Combine with motion mask if available
        if motion_mask is not None and self.enable_motion_detection:
            # Dilate motion mask to create regions around motion
            kernel = np.ones((15, 15), np.uint8)
            dilated_motion = cv2.dilate(motion_mask, kernel, iterations=1)
            
            # Combine ROI mask with motion mask (AND operation)
            mask = cv2.bitwise_and(mask, dilated_motion)
        
        # Apply mask to frame
        processed_frame = frame.copy()
        processed_frame[mask == 0] = 0
        
        metadata["roi_applied"] = True
        
        return processed_frame, metadata
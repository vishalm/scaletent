"""
Camera management module for ScaleTent
"""

import asyncio
import cv2
import numpy as np
import time
import threading
from pathlib import Path
import queue

from src.core.logger import setup_logger

logger = setup_logger(__name__)

class Camera:
    """
    Camera class for handling video input
    """
    
    def __init__(self, camera_id, source, width=1280, height=720, fps=30):
        """
        Initialize camera
        
        Args:
            camera_id (str): Unique camera identifier
            source (str): Camera source (URL, device ID, file path)
            width (int): Output frame width
            height (int): Output frame height
            fps (int): Target frames per second
        """
        self.camera_id = camera_id
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        
        # Frame buffer and stats
        self.frame_buffer = queue.Queue(maxsize=5)
        self.frame_count = 0
        self.last_frame_time = 0
        self.fps = 0
        
        # Camera state
        self.is_running = False
        self.capture_thread = None
        self.last_error = None
        self.capture = None
        
        logger.info(f"Camera {camera_id} initialized with source: {source}")
    
    def start(self):
        """Start camera capture thread"""
        if self.is_running:
            logger.warning(f"Camera {self.camera_id} is already running")
            return
        
        # Start capture thread
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        logger.info(f"Camera {self.camera_id} started")
    
    def stop(self):
        """Stop camera capture thread"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        logger.info(f"Camera {self.camera_id} stopped")
    
    def get_frame(self):
        """
        Get the latest frame from the camera
        
        Returns:
            tuple: (frame, timestamp)
        """
        try:
            return self.frame_buffer.get(block=False)
        except queue.Empty:
            return None, 0
    
    def _capture_loop(self):
        """Camera capture loop (runs in a separate thread)"""
        retry_count = 0
        max_retries = 5
        retry_delay = 2.0  # seconds
        
        while self.is_running:
            try:
                # Open capture if needed
                if self.capture is None:
                    self._open_capture()
                    # Skip rest of loop if open failed
                    if self.capture is None:
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.error(f"Failed to open camera {self.camera_id} after {max_retries} attempts")
                            self.is_running = False
                            return
                        
                        logger.info(f"Retrying camera {self.camera_id} in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                
                # Read frame
                ret, frame = self.capture.read()
                
                if not ret or frame is None:
                    logger.warning(f"Failed to read frame from camera {self.camera_id}")
                    self.capture.release()
                    self.capture = None
                    continue
                
                # Resize frame if needed
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Calculate FPS
                current_time = time.time()
                if self.last_frame_time > 0:
                    time_diff = current_time - self.last_frame_time
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)  # Smoothed FPS
                self.last_frame_time = current_time
                
                # Update frame count
                self.frame_count += 1
                
                # Add frame to buffer (remove oldest if full)
                if self.frame_buffer.full():
                    try:
                        self.frame_buffer.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_buffer.put((frame, current_time))
                
                # Control capture rate to match target FPS
                if self.target_fps > 0:
                    target_time_per_frame = 1.0 / self.target_fps
                    elapsed = time.time() - current_time
                    sleep_time = max(0, target_time_per_frame - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Reset retry count on successful frame capture
                retry_count = 0
                
            except Exception as e:
                logger.error(f"Error in camera {self.camera_id} capture loop: {e}")
                self.last_error = str(e)
                
                # Release and retry
                if self.capture:
                    self.capture.release()
                    self.capture = None
                
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Too many errors in camera {self.camera_id}, stopping")
                    self.is_running = False
                    return
                
                time.sleep(retry_delay)
    
    def _open_capture(self):
        """Open the camera capture"""
        try:
            # Interpret source (int device ID, str for URL/path)
            if isinstance(self.source, str) and self.source.isdigit():
                source = int(self.source)
            else:
                source = self.source
            
            logger.info(f"Opening camera {self.camera_id} with source: {source}")
            self.capture = cv2.VideoCapture(source)
            
            # Check if opened successfully
            if not self.capture.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                self.capture = None
                return
            
            # Set properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Try to set FPS (may not be supported by all cameras)
            if self.target_fps > 0:
                self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Read actual properties
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.capture.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera {self.camera_id} opened successfully")
            logger.info(f"Resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
        except Exception as e:
            logger.error(f"Error opening camera {self.camera_id}: {e}")
            self.last_error = str(e)
            self.capture = None


class CameraManager:
    """
    Manager for multiple cameras
    """
    
    def __init__(self, config, detector, face_detector, face_matcher, publisher):
        """
        Initialize camera manager
        
        Args:
            config: Configuration object
            detector: Object detector
            face_detector: Face detector
            face_matcher: Face matcher
            publisher: Stream publisher
        """
        self.config = config
        self.detector = detector
        self.face_detector = face_detector
        self.face_matcher = face_matcher
        self.publisher = publisher
        
        # Camera instances
        self.cameras = {}
        
        # Processing variables
        self.processing_tasks = {}
        self.is_running = False
        
        logger.info("Camera manager initialized")
    
    async def start(self):
        """Start all cameras and processing tasks"""
        if self.is_running:
            logger.warning("Camera manager is already running")
            return
        
        self.is_running = True
        
        # Configure cameras from config
        camera_configs = self.config.get("cameras", [])
        for camera_config in camera_configs:
            camera_id = camera_config.get("id")
            source = camera_config.get("source")
            
            if not camera_id or not source:
                logger.error("Camera config missing id or source")
                continue
            
            # Create camera instance
            camera = Camera(
                camera_id=camera_id,
                source=source,
                width=camera_config.get("width", 1280),
                height=camera_config.get("height", 720),
                fps=camera_config.get("fps", 30)
            )
            
            # Add to cameras dict
            self.cameras[camera_id] = camera
            
            # Start camera
            camera.start()
            
            # Start processing task
            task = asyncio.create_task(self._process_camera(camera_id))
            self.processing_tasks[camera_id] = task
        
        logger.info(f"Started {len(self.cameras)} cameras")
    
    async def stop(self):
        """Stop all cameras and processing tasks"""
        self.is_running = False
        
        # Cancel all processing tasks
        for camera_id, task in self.processing_tasks.items():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
        
        self.processing_tasks = {}
        
        # Stop all cameras
        for camera_id, camera in self.cameras.items():
            camera.stop()
        
        self.cameras = {}
        
        logger.info("Camera manager stopped")
    
    async def _process_camera(self, camera_id):
        """
        Process frames from a camera
        
        Args:
            camera_id (str): Camera ID to process
        """
        logger.info(f"Starting processing for camera {camera_id}")
        
        camera = self.cameras.get(camera_id)
        if not camera:
            logger.error(f"Camera {camera_id} not found in manager")
            return
        
        processing_interval = 1.0 / self.config.get(f"cameras.{camera_id}.processing_fps", 10)
        last_process_time = 0
        
        while self.is_running:
            try:
                # Check if time to process frame
                current_time = time.time()
                if current_time - last_process_time < processing_interval:
                    await asyncio.sleep(0.01)  # Small sleep to prevent CPU hogging
                    continue
                
                # Get latest frame
                frame, timestamp = camera.get_frame()
                
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process frame
                await self._process_frame(camera_id, frame, timestamp)
                
                # Update last process time
                last_process_time = time.time()
                
            except asyncio.CancelledError:
                logger.info(f"Processing task for camera {camera_id} cancelled")
                break
                
            except Exception as e:
                logger.error(f"Error processing camera {camera_id}: {e}")
                await asyncio.sleep(1.0)  # Delay before retry
        
        logger.info(f"Stopped processing for camera {camera_id}")
    
    async def _process_frame(self, camera_id, frame, timestamp):
        """
        Process a single frame
        
        Args:
            camera_id (str): Camera ID
            frame (numpy.ndarray): Input frame
            timestamp (float): Frame timestamp
        """
        try:
            # Detect people
            detections = self.detector.detect(frame)
            
            # Detect faces
            face_detections = self.face_detector.detect_faces(frame, detections)
            
            # Match faces
            for face in face_detections:
                recognized, identity_id, similarity, person_data = self.face_matcher.match_face_in_frame(frame, face)
                
                # Link face to person detection
                if "person_id" in face:
                    person_id = face["person_id"]
                    
                    # Find the person detection
                    for detection in detections:
                        if detection["id"] == person_id:
                            # Add recognition info
                            detection["recognized"] = recognized
                            if recognized and person_data:
                                detection["person_data"] = person_data
                            break
            
            # Prepare data for publishing
            data = {
                "frame_id": int(timestamp * 1000),
                "camera_id": camera_id,
                "timestamp": timestamp,
                "detections": detections
            }
            
            # Publish data
            await self.publisher.publish(data)
            
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
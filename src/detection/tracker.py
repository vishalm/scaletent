"""
Object tracking module for ScaleTent
Handles tracking of detected objects across frames
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict

from core.logger import setup_logger

logger = setup_logger(__name__)

class Detection:
    """Class representing a single detection"""
    def __init__(self, bbox, confidence=0.0, class_id=0, class_name="person"):
        """
        Initialize a detection
        
        Args:
            bbox (list): Bounding box [x1, y1, x2, y2]
            confidence (float): Detection confidence
            class_id (int): Class ID
            class_name (str): Class name
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        
        # Calculate box centroid
        self.centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        # Calculate box area
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


class TrackedObject:
    """Class representing a tracked object"""
    id_counter = 0
    
    def __init__(self, detection):
        """
        Initialize a tracked object
        
        Args:
            detection (Detection): Initial detection
        """
        self.id = f"object-{TrackedObject.id_counter}"
        TrackedObject.id_counter += 1
        
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.centroid = detection.centroid
        
        # Tracking state
        self.age = 0
        self.consecutive_missed = 0
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.lost = False
        
        # Path history
        self.path = [self.centroid]
        self.max_path_length = 50
        
        # Appearance history
        self.bboxes = [self.bbox]
        self.confidences = [self.confidence]
        self.max_history = 10

    def update(self, detection):
        """
        Update tracked object with new detection
        
        Args:
            detection (Detection): New detection
        """
        # Update bounding box history
        self.bboxes.append(detection.bbox)
        if len(self.bboxes) > self.max_history:
            self.bboxes = self.bboxes[-self.max_history:]
        
        # Update confidence history
        self.confidences.append(detection.confidence)
        if len(self.confidences) > self.max_history:
            self.confidences = self.confidences[-self.max_history:]
        
        # Update current state
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.centroid = detection.centroid
        
        # Update path history
        self.path.append(self.centroid)
        if len(self.path) > self.max_path_length:
            self.path = self.path[-self.max_path_length:]
        
        # Update tracking state
        self.age += 1
        self.consecutive_missed = 0
        self.last_seen = time.time()
        self.lost = False
    
    def mark_missed(self):
        """Mark object as missed in current frame"""
        self.consecutive_missed += 1
        
        # Mark as lost if missed for too long
        if self.consecutive_missed >= 10:
            self.lost = True
    
    def get_average_bbox(self):
        """
        Get the average bounding box of last N frames
        
        Returns:
            list: Average bounding box [x1, y1, x2, y2]
        """
        if not self.bboxes:
            return [0, 0, 0, 0]
        
        # Calculate average for each coordinate
        avg_bbox = [0, 0, 0, 0]
        for bbox in self.bboxes:
            for i in range(4):
                avg_bbox[i] += bbox[i]
        
        for i in range(4):
            avg_bbox[i] = int(avg_bbox[i] / len(self.bboxes))
        
        return avg_bbox
    
    def get_average_confidence(self):
        """
        Get the average confidence of last N frames
        
        Returns:
            float: Average confidence
        """
        if not self.confidences:
            return 0.0
        
        return sum(self.confidences) / len(self.confidences)
    
    def predict_next_position(self):
        """
        Predict next position based on motion history
        
        Returns:
            tuple: Predicted (x, y) centroid
        """
        # Need at least 2 points for prediction
        if len(self.path) < 2:
            return self.centroid
        
        # Simple linear prediction
        last_point = self.path[-1]
        prev_point = self.path[-2]
        
        dx = last_point[0] - prev_point[0]
        dy = last_point[1] - prev_point[1]
        
        predicted_x = last_point[0] + dx
        predicted_y = last_point[1] + dy
        
        return (predicted_x, predicted_y)

    def to_dict(self):
        """
        Convert to dictionary representation
        
        Returns:
            dict: Dictionary representation
        """
        return {
            "id": self.id,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "confidence": self.get_average_confidence(),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "age": self.age,
            "time_visible": time.time() - self.first_seen,
            "last_seen": time.time() - self.last_seen,
            "lost": self.lost
        }


class ObjectTracker:
    """
    Multiple object tracker for ScaleTent
    
    Supports multiple tracking algorithms:
    1. SORT (Simple Online and Realtime Tracking)
    2. IoU Tracker (Intersection over Union)
    3. Centroid Tracker
    4. OpenCV Trackers (KCF, CSRT, etc.)
    5. DeepSORT (with appearance features)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tracker
        
        Args:
            config (dict): Tracker configuration
        """
        self.config = config
        
        # Tracking algorithm
        self.algorithm = config.get("algorithm", "iou")
        
        # Tracking parameters
        self.max_disappeared = config.get("max_disappeared", 30)
        self.max_age = config.get("max_age", 100)
        self.min_hits = config.get("min_hits", 3)
        self.iou_threshold = config.get("iou_threshold", 0.3)
        
        # Tracked objects dict (id -> TrackedObject)
        self.objects = OrderedDict()
        
        # Frame dimensions for boundary checks
        self.frame_width = 0
        self.frame_height = 0
        
        # External tracker instances (for OpenCV trackers)
        self.cv_trackers = {}
        
        # Initialize algorithm-specific components
        self._init_algorithm()
        
        logger.info(f"Object tracker initialized with algorithm: {self.algorithm}")
    
    def _init_algorithm(self):
        """Initialize algorithm-specific components"""
        if self.algorithm == "sort":
            try:
                # Try to import the SORT tracker
                from sort.sort import Sort
                
                self.sort_tracker = Sort(
                    max_age=self.max_age,
                    min_hits=self.min_hits,
                    iou_threshold=self.iou_threshold
                )
                logger.info("SORT tracker initialized")
            except ImportError:
                logger.warning("SORT tracker not available, falling back to IoU tracker")
                self.algorithm = "iou"
        
        elif self.algorithm == "deep_sort":
            try:
                # Try to import the DeepSORT tracker
                from deep_sort.deep_sort import DeepSort
                
                model_path = self.config.get("appearance_model", "models/mars-small128.pb")
                self.deep_sort_tracker = DeepSort(model_path)
                logger.info("DeepSORT tracker initialized")
            except ImportError:
                logger.warning("DeepSORT tracker not available, falling back to IoU tracker")
                self.algorithm = "iou"
        
        elif self.algorithm == "opencv":
            # OpenCV tracker type
            self.cv_tracker_type = self.config.get("cv_tracker_type", "KCF")
            logger.info(f"OpenCV {self.cv_tracker_type} tracker will be used")
    
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of detection dictionaries
        
        Returns:
            list: Updated list of tracked objects
        """
        # Update frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Convert detection dictionaries to Detection objects
        detection_objects = []
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            confidence = det.get('confidence', 0.0)
            class_id = det.get('class_id', 0)
            class_name = det.get('class_name', 'person')
            detection_objects.append(Detection(bbox, confidence, class_id, class_name))
        
        # Choose the appropriate tracking algorithm
        if self.algorithm == "sort":
            return self._update_sort(frame, detection_objects)
        elif self.algorithm == "deep_sort":
            return self._update_deep_sort(frame, detection_objects)
        elif self.algorithm == "opencv":
            return self._update_opencv(frame, detection_objects)
        elif self.algorithm == "centroid":
            return self._update_centroid(frame, detection_objects)
        else:
            # Default to IoU tracker
            return self._update_iou(frame, detection_objects)
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            bbox1 (list): First bounding box [x1, y1, x2, y2]
            bbox2 (list): Second bounding box [x1, y1, x2, y2]
        
        Returns:
            float: IoU score
        """
        # Determine intersection coordinates
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # Check if there is no intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate union area
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def _update_iou(self, frame: np.ndarray, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Update using IoU tracking method
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of Detection objects
        
        Returns:
            list: Updated list of tracked objects
        """
        # If no objects are being tracked yet, initialize with first detections
        if len(self.objects) == 0:
            for detection in detections:
                self.objects[f"object-{TrackedObject.id_counter}"] = TrackedObject(detection)
            
            # Return current objects
            return [obj.to_dict() for obj in self.objects.values()]
        
        # Create lists of tracked and new centroids
        tracked_objects = list(self.objects.values())
        tracked_bboxes = [obj.bbox for obj in tracked_objects]
        
        # Mark all current objects as not updated yet
        updated_objects = {obj_id: False for obj_id in self.objects}
        
        # Match detections to existing objects based on IoU
        for detection in detections:
            # Calculate IoU with all existing objects
            ious = [self._calculate_iou(detection.bbox, obj_bbox) for obj_bbox in tracked_bboxes]
            
            # Find the best match
            if ious and max(ious) >= self.iou_threshold:
                # Get the index of the best match
                match_idx = np.argmax(ious)
                obj_id = list(self.objects.keys())[match_idx]
                
                # Update the matched object
                self.objects[obj_id].update(detection)
                updated_objects[obj_id] = True
            else:
                # Create a new tracked object
                new_obj = TrackedObject(detection)
                self.objects[new_obj.id] = new_obj
                updated_objects[new_obj.id] = True
        
        # Handle disappeared objects
        # Make a copy of keys to avoid modification during iteration
        object_ids = list(self.objects.keys())
        for obj_id in object_ids:
            if not updated_objects.get(obj_id, False):
                self.objects[obj_id].mark_missed()
                
                # Remove objects that have been missed for too long
                if self.objects[obj_id].consecutive_missed > self.max_disappeared:
                    del self.objects[obj_id]
        
        # Return current objects
        return [obj.to_dict() for obj in self.objects.values() if not obj.lost]
    
    def _update_centroid(self, frame: np.ndarray, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Update using centroid tracking method
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of Detection objects
        
        Returns:
            list: Updated list of tracked objects
        """
        # If no objects are being tracked yet, initialize with first detections
        if len(self.objects) == 0:
            for detection in detections:
                self.objects[f"object-{TrackedObject.id_counter}"] = TrackedObject(detection)
            
            # Return current objects
            return [obj.to_dict() for obj in self.objects.values()]
        
        # Create lists of tracked and new centroids
        tracked_objects = list(self.objects.values())
        tracked_centroids = [obj.centroid for obj in tracked_objects]
        new_centroids = [detection.centroid for detection in detections]
        
        # If no new detections, mark all existing objects as disappeared
        if len(new_centroids) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].mark_missed()
                
                # Remove objects that have been missed for too long
                if self.objects[obj_id].consecutive_missed > self.max_disappeared:
                    del self.objects[obj_id]
            
            # Return current objects
            return [obj.to_dict() for obj in self.objects.values() if not obj.lost]
        
        # If no existing objects, create new ones
        if len(tracked_centroids) == 0:
            for detection in detections:
                self.objects[f"object-{TrackedObject.id_counter}"] = TrackedObject(detection)
        
        # Calculate distances between each pair of existing and new centroids
        distances = np.zeros((len(tracked_centroids), len(new_centroids)))
        for i, tracked_centroid in enumerate(tracked_centroids):
            for j, new_centroid in enumerate(new_centroids):
                # Calculate Euclidean distance
                d = np.sqrt((tracked_centroid[0] - new_centroid[0])**2 + 
                            (tracked_centroid[1] - new_centroid[1])**2)
                distances[i, j] = d
        
        # Find the smallest distances and associate objects accordingly
        # Using the Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(distances)
        except ImportError:
            # Fallback to greedy assignment if scipy not available
            row_indices = []
            col_indices = []
            
            # Copy the distance matrix to modify
            dist_copy = distances.copy()
            
            # While there are still rows and columns to process
            while dist_copy.size > 0 and dist_copy.shape[0] > 0 and dist_copy.shape[1] > 0:
                # Find the minimum distance
                min_idx = np.unravel_index(np.argmin(dist_copy), dist_copy.shape)
                
                # Add to our matches if the distance is below threshold
                if dist_copy[min_idx] < 100:  # Max distance threshold
                    row_indices.append(min_idx[0])
                    col_indices.append(min_idx[1])
                
                # Remove the row and column
                dist_copy = np.delete(dist_copy, min_idx[0], axis=0)
                dist_copy = np.delete(dist_copy, min_idx[1], axis=1)
        
        # Mark all current objects as not updated yet
        updated_objects = {obj_id: False for obj_id in self.objects}
        
        # Update matched objects
        for row_idx, col_idx in zip(row_indices, col_indices):
            # Only update if the distance is reasonable
            if distances[row_idx, col_idx] < 100:  # Max distance threshold
                # Get the object ID
                obj_id = list(self.objects.keys())[row_idx]
                
                # Update the matched object
                self.objects[obj_id].update(detections[col_idx])
                updated_objects[obj_id] = True
        
        # Add new objects for unmatched detections
        unmatched_cols = set(range(len(new_centroids))) - set(col_indices)
        for col_idx in unmatched_cols:
            # Create a new tracked object
            new_obj = TrackedObject(detections[col_idx])
            self.objects[new_obj.id] = new_obj
        
        # Handle disappeared objects
        # Make a copy of keys to avoid modification during iteration
        object_ids = list(self.objects.keys())
        for obj_id in object_ids:
            if not updated_objects.get(obj_id, False):
                self.objects[obj_id].mark_missed()
                
                # Remove objects that have been missed for too long
                if self.objects[obj_id].consecutive_missed > self.max_disappeared:
                    del self.objects[obj_id]
        
        # Return current objects
        return [obj.to_dict() for obj in self.objects.values() if not obj.lost]
    
    def _update_sort(self, frame: np.ndarray, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Update using SORT tracking method
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of Detection objects
        
        Returns:
            list: Updated list of tracked objects
        """
        # Convert detections to SORT format: [x1, y1, x2, y2, confidence]
        if not detections:
            # No detections, update with empty detections
            if hasattr(self, 'sort_tracker'):
                track_bbs_ids = self.sort_tracker.update(np.empty((0, 5)))
                
                # Clear existing objects that aren't in the current tracks
                current_ids = set()
                
                # Update existing objects and create new ones
                for x1, y1, x2, y2, obj_id in track_bbs_ids:
                    obj_id = int(obj_id)
                    current_ids.add(obj_id)
                
                # Remove objects that are no longer tracked
                self.objects = {k: v for k, v in self.objects.items() if int(k.split('-')[-1]) in current_ids}
                
                # Return current objects
                return [obj.to_dict() for obj in self.objects.values()]
            else:
                return []
        
        detection_array = np.array([[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence] for d in detections])
        
        # Update SORT tracker
        track_bbs_ids = self.sort_tracker.update(detection_array)
        
        # Clear existing objects that aren't in the current tracks
        current_ids = set()
        
        # Update existing objects and create new ones
        for x1, y1, x2, y2, obj_id in track_bbs_ids:
            obj_id = int(obj_id)
            current_ids.add(obj_id)
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            
            # Find corresponding detection for confidence
            confidence = 0.0
            class_id = 0
            class_name = "person"
            
            for det in detections:
                # Use IoU to find the best match
                iou = self._calculate_iou(bbox, det.bbox)
                if iou > 0.5:  # Threshold for matching
                    confidence = det.confidence
                    class_id = det.class_id
                    class_name = det.class_name
                    break
            
            # Create mock detection for update
            detection = Detection(bbox, confidence, class_id, class_name)
            
            # Check if this object is already being tracked
            obj_key = f"object-{obj_id}"
            if obj_key in self.objects:
                # Update existing object
                self.objects[obj_key].update(detection)
            else:
                # Create new tracked object (with SORT ID)
                new_obj = TrackedObject(detection)
                new_obj.id = obj_key
                self.objects[obj_key] = new_obj
        
        # Remove objects that are no longer tracked
        self.objects = {k: v for k, v in self.objects.items() if int(k.split('-')[-1]) in current_ids}
        
        # Return current objects
        return [obj.to_dict() for obj in self.objects.values()]
    
    def _update_deep_sort(self, frame: np.ndarray, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Update using DeepSORT tracking method
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of Detection objects
        
        Returns:
            list: Updated list of tracked objects
        """
        # Convert detections to DeepSORT format: [x1, y1, x2, y2, confidence]
        if not detections:
            return []
        
        xywhs = []
        confidences = []
        class_ids = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            x_center = x1 + w / 2
            y_center = y1 + h / 2
            
            xywhs.append([x_center, y_center, w, h])
            confidences.append(det.confidence)
            class_ids.append(det.class_id)
        
        xywhs = torch.Tensor(xywhs)
        confidences = torch.Tensor(confidences)
        
        # Update DeepSORT tracker
        outputs = self.deep_sort_tracker.update(xywhs, confidences, class_ids, frame)
        
        # Clear existing objects that aren't in the current tracks
        current_ids = set()
        
        # List to store updated tracked objects
        updated_objects = []
        
        # Update existing objects and create new ones
        for output in outputs:
            x1, y1, x2, y2, track_id, class_id, confidence = output
            
            current_ids.add(track_id)
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            
            # Find class name from class_id
            class_name = "person"  # Default
            for det in detections:
                if det.class_id == class_id:
                    class_name = det.class_name
                    break
            
            # Create mock detection for update
            detection = Detection(bbox, confidence, class_id, class_name)
            
            # Check if this object is already being tracked
            obj_key = f"object-{track_id}"
            if obj_key in self.objects:
                # Update existing object
                self.objects[obj_key].update(detection)
            else:
                # Create new tracked object (with DeepSORT ID)
                new_obj = TrackedObject(detection)
                new_obj.id = obj_key
                self.objects[obj_key] = new_obj
            
            updated_objects.append(self.objects[obj_key].to_dict())
        
        # Remove objects that are no longer tracked
        self.objects = {k: v for k, v in self.objects.items() if int(k.split('-')[-1]) in current_ids}
        
        return updated_objects
    
    def _update_opencv(self, frame: np.ndarray, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Update using OpenCV tracking methods
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of Detection objects
        
        Returns:
            list: Updated list of tracked objects
        """
        # Create trackers for new detections
        for detection in detections:
            # Check if this detection can be matched to existing object
            matched = False
            for obj_id, obj in list(self.objects.items()):
                # Use IoU to check if this is the same object
                if self._calculate_iou(detection.bbox, obj.bbox) > self.iou_threshold:
                    # Update existing object
                    obj.update(detection)
                    matched = True
                    
                    # Reset the tracker for this object
                    self._create_tracker(obj_id, frame, detection.bbox)
                    break
            
            if not matched:
                # Create new object
                new_obj = TrackedObject(detection)
                self.objects[new_obj.id] = new_obj
                
                # Create a tracker for the new object
                self._create_tracker(new_obj.id, frame, detection.bbox)
        
        # Update objects with trackers
        for obj_id in list(self.objects.keys()):
            if obj_id in self.cv_trackers:
                # Update tracker
                success, bbox = self.cv_trackers[obj_id].update(frame)
                
                if success:
                    # Format as [x1, y1, x2, y2]
                    x, y, w, h = list(map(int, bbox))
                    bbox = [x, y, x + w, y + h]
                    
                    # Update object with tracker result
                    detection = Detection(bbox, self.objects[obj_id].confidence)
                    self.objects[obj_id].update(detection)
                else:
                    # Tracker lost the object
                    self.objects[obj_id].mark_missed()
                    del self.cv_trackers[obj_id]
            else:
                # No tracker for this object
                self.objects[obj_id].mark_missed()
            
            # Remove lost objects
            if self.objects[obj_id].consecutive_missed > self.max_disappeared:
                del self.objects[obj_id]
                if obj_id in self.cv_trackers:
                    del self.cv_trackers[obj_id]
        
        # Return current objects
        return [obj.to_dict() for obj in self.objects.values() if not obj.lost]
    
    def _create_tracker(self, obj_id: str, frame: np.ndarray, bbox: List[int]):
        """
        Create a new OpenCV tracker
        
        Args:
            obj_id (str): Object ID
            frame (numpy.ndarray): Current frame
            bbox (list): Bounding box [x1, y1, x2, y2]
        """
        # Convert bbox format from [x1, y1, x2, y2] to [x, y, w, h]
        x1, y1, x2, y2 = bbox
        cv_bbox = (x1, y1, x2 - x1, y2 - y1)
        
        # Create appropriate tracker
        if self.cv_tracker_type == "KCF":
            tracker = cv2.TrackerKCF_create()
        elif self.cv_tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        elif self.cv_tracker_type == "MOSSE":
            tracker = cv2.legacy.TrackerMOSSE_create()
        elif self.cv_tracker_type == "MIL":
            tracker = cv2.TrackerMIL_create()
        else:
            # Default to KCF
            tracker = cv2.TrackerKCF_create()
        
        # Initialize tracker
        tracker.init(frame, cv_bbox)
        
        # Store tracker
        self.cv_trackers[obj_id] = tracker
    
    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tracks on frame
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            numpy.ndarray: Frame with tracks drawn
        """
        # Make a copy of the frame
        output = frame.copy()
        
        # Draw each tracked object
        for obj_id, obj in self.objects.items():
            if obj.lost:
                continue
            
            # Get bounding box
            x1, y1, x2, y2 = obj.bbox
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID
            cv2.putText(output, obj_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
            
            # Draw path
            if len(obj.path) > 1:
                # Convert to numpy array
                path = np.array(obj.path)
                
                # Draw path line
                for i in range(1, len(path"""
Object tracking module for ScaleTent
Handles tracking of detected objects across frames
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict

from core.logger import setup_logger

logger = setup_logger(__name__)

class Detection:
    """Class representing a single detection"""
    def __init__(self, bbox, confidence=0.0, class_id=0, class_name="person"):
        """
        Initialize a detection
        
        Args:
            bbox (list): Bounding box [x1, y1, x2, y2]
            confidence (float): Detection confidence
            class_id (int): Class ID
            class_name (str): Class name
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        
        # Calculate box centroid
        self.centroid = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        # Calculate box area
        self.area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


class TrackedObject:
    """Class representing a tracked object"""
    id_counter = 0
    
    def __init__(self, detection):
        """
        Initialize a tracked object
        
        Args:
            detection (Detection): Initial detection
        """
        self.id = f"object-{TrackedObject.id_counter}"
        TrackedObject.id_counter += 1
        
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.centroid = detection.centroid
        
        # Tracking state
        self.age = 0
        self.consecutive_missed = 0
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.lost = False
        
        # Path history
        self.path = [self.centroid]
        self.max_path_length = 50
        
        # Appearance history
        self.bboxes = [self.bbox]
        self.confidences = [self.confidence]
        self.max_history = 10

    def update(self, detection):
        """
        Update tracked object with new detection
        
        Args:
            detection (Detection): New detection
        """
        # Update bounding box history
        self.bboxes.append(detection.bbox)
        if len(self.bboxes) > self.max_history:
            self.bboxes = self.bboxes[-self.max_history:]
        
        # Update confidence history
        self.confidences.append(detection.confidence)
        if len(self.confidences) > self.max_history:
            self.confidences = self.confidences[-self.max_history:]
        
        # Update current state
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.centroid = detection.centroid
        
        # Update path history
        self.path.append(self.centroid)
        if len(self.path) > self.max_path_length:
            self.path = self.path[-self.max_path_length:]
        
        # Update tracking state
        self.age += 1
        self.consecutive_missed = 0
        self.last_seen = time.time()
        self.lost = False
    
    def mark_missed(self):
        """Mark object as missed in current frame"""
        self.consecutive_missed += 1
        
        # Mark as lost if missed for too long
        if self.consecutive_missed >= 10:
            self.lost = True
    
    def get_average_bbox(self):
        """
        Get the average bounding box of last N frames
        
        Returns:
            list: Average bounding box [x1, y1, x2, y2]
        """
        if not self.bboxes:
            return [0, 0, 0, 0]
        
        # Calculate average for each coordinate
        avg_bbox = [0, 0, 0, 0]
        for bbox in self.bboxes:
            for i in range(4):
                avg_bbox[i] += bbox[i]
        
        for i in range(4):
            avg_bbox[i] = int(avg_bbox[i] / len(self.bboxes))
        
        return avg_bbox
    
    def get_average_confidence(self):
        """
        Get the average confidence of last N frames
        
        Returns:
            float: Average confidence
        """
        if not self.confidences:
            return 0.0
        
        return sum(self.confidences) / len(self.confidences)
    
    def predict_next_position(self):
        """
        Predict next position based on motion history
        
        Returns:
            tuple: Predicted (x, y) centroid
        """
        # Need at least 2 points for prediction
        if len(self.path) < 2:
            return self.centroid
        
        # Simple linear prediction
        last_point = self.path[-1]
        prev_point = self.path[-2]
        
        dx = last_point[0] - prev_point[0]
        dy = last_point[1] - prev_point[1]
        
        predicted_x = last_point[0] + dx
        predicted_y = last_point[1] + dy
        
        return (predicted_x, predicted_y)

    def to_dict(self):
        """
        Convert to dictionary representation
        
        Returns:
            dict: Dictionary representation
        """
        return {
            "id": self.id,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "confidence": self.get_average_confidence(),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "age": self.age,
            "time_visible": time.time() - self.first_seen,
            "last_seen": time.time() - self.last_seen,
            "lost": self.lost
        }


class ObjectTracker:
    """
    Multiple object tracker for ScaleTent
    
    Supports multiple tracking algorithms:
    1. SORT (Simple Online and Realtime Tracking)
    2. IoU Tracker (Intersection over Union)
    3. Centroid Tracker
    4. OpenCV Trackers (KCF, CSRT, etc.)
    5. DeepSORT (with appearance features)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tracker
        
        Args:
            config (dict): Tracker configuration
        """
        self.config = config
        
        # Tracking algorithm
        self.algorithm = config.get("algorithm", "iou")
        
        # Tracking parameters
        self.max_disappeared = config.get("max_disappeared", 30)
        self.max_age = config.get("max_age", 100)
        self.min_hits = config.get("min_hits", 3)
        self.iou_threshold = config.get("iou_threshold", 0.3)
        
        # Tracked objects dict (id -> TrackedObject)
        self.objects = OrderedDict()
        
        # Frame dimensions for boundary checks
        self.frame_width = 0
        self.frame_height = 0
        
        # External tracker instances (for OpenCV trackers)
        self.cv_trackers = {}
        
        # Initialize algorithm-specific components
        self._init_algorithm()
        
        logger.info(f"Object tracker initialized with algorithm: {self.algorithm}")
    
    def _init_algorithm(self):
        """Initialize algorithm-specific components"""
        if self.algorithm == "sort":
            try:
                # Try to import the SORT tracker
                from sort.sort import Sort
                
                self.sort_tracker = Sort(
                    max_age=self.max_age,
                    min_hits=self.min_hits,
                    iou_threshold=self.iou_threshold
                )
                logger.info("SORT tracker initialized")
            except ImportError:
                logger.warning("SORT tracker not available, falling back to IoU tracker")
                self.algorithm = "iou"
        
        elif self.algorithm == "deep_sort":
            try:
                # Try to import the DeepSORT tracker
                from deep_sort.deep_sort import DeepSort
                
                model_path = self.config.get("appearance_model", "models/mars-small128.pb")
                self.deep_sort_tracker = DeepSort(model_path)
                logger.info("DeepSORT tracker initialized")
            except ImportError:
                logger.warning("DeepSORT tracker not available, falling back to IoU tracker")
                self.algorithm = "iou"
        
        elif self.algorithm == "opencv":
            # OpenCV tracker type
            self.cv_tracker_type = self.config.get("cv_tracker_type", "KCF")
            logger.info(f"OpenCV {self.cv_tracker_type} tracker will be used")
    
    def update(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of detection dictionaries
        
        Returns:
            list: Updated list of tracked objects
        """
        # Update frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Convert detection dictionaries to Detection objects
        detection_objects = []
        for det in detections:
            bbox = det.get('bbox', [0, 0, 0, 0])
            confidence = det.get('confidence', 0.0)
            class_id = det.get('class_id', 0)
            class_name = det.get('class_name', 'person')
            detection_objects.append(Detection(bbox, confidence, class_id, class_name))
        
        # Choose the appropriate tracking algorithm
        if self.algorithm == "sort":
            return self._update_sort(frame, detection_objects)
        elif self.algorithm == "deep_sort":
            return self._update_deep_sort(frame, detection_objects)
        elif self.algorithm == "opencv":
            return self._update_opencv(frame, detection_objects)
        elif self.algorithm == "centroid":
            return self._update_centroid(frame, detection_objects)
        else:
            # Default to IoU tracker
            return self._update_iou(frame, detection_objects)
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            bbox1 (list): First bounding box [x1, y1, x2, y2]
            bbox2 (list): Second bounding box [x1, y1, x2, y2]
        
        Returns:
            float: IoU score
        """
        # Determine intersection coordinates
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # Check if there is no intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate union area
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def _update_iou(self, frame: np.ndarray, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Update using IoU tracking method
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of Detection objects
        
        Returns:
            list: Updated list of tracked objects
        """
        # If no objects are being tracked yet, initialize with first detections
        if len(self.objects) == 0:
            for detection in detections:
                self.objects[f"object-{TrackedObject.id_counter}"] = TrackedObject(detection)
            
            # Return current objects
            return [obj.to_dict() for obj in self.objects.values()]
        
        # Create lists of tracked and new centroids
        tracked_objects = list(self.objects.values())
        tracked_bboxes = [obj.bbox for obj in tracked_objects]
        
        # Mark all current objects as not updated yet
        updated_objects = {obj_id: False for obj_id in self.objects}
        
        # Match detections to existing objects based on IoU
        for detection in detections:
            # Calculate IoU with all existing objects
            ious = [self._calculate_iou(detection.bbox, obj_bbox) for obj_bbox in tracked_bboxes]
            
            # Find the best match
            if ious and max(ious) >= self.iou_threshold:
                # Get the index of the best match
                match_idx = np.argmax(ious)
                obj_id = list(self.objects.keys())[match_idx]
                
                # Update the matched object
                self.objects[obj_id].update(detection)
                updated_objects[obj_id] = True
            else:
                # Create a new tracked object
                new_obj = TrackedObject(detection)
                self.objects[new_obj.id] = new_obj
                updated_objects[new_obj.id] = True
        
        # Handle disappeared objects
        # Make a copy of keys to avoid modification during iteration
        object_ids = list(self.objects.keys())
        for obj_id in object_ids:
            if not updated_objects.get(obj_id, False):
                self.objects[obj_id].mark_missed()
                
                # Remove objects that have been missed for too long
                if self.objects[obj_id].consecutive_missed > self.max_disappeared:
                    del self.objects[obj_id]
        
        # Return current objects
        return [obj.to_dict() for obj in self.objects.values() if not obj.lost]
    
    def _update_centroid(self, frame: np.ndarray, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Update using centroid tracking method
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of Detection objects
        
        Returns:
            list: Updated list of tracked objects
        """
        # If no objects are being tracked yet, initialize with first detections
        if len(self.objects) == 0:
            for detection in detections:
                self.objects[f"object-{TrackedObject.id_counter}"] = TrackedObject(detection)
            
            # Return current objects
            return [obj.to_dict() for obj in self.objects.values()]
        
        # Create lists of tracked and new centroids
        tracked_objects = list(self.objects.values())
        tracked_centroids = [obj.centroid for obj in tracked_objects]
        new_centroids = [detection.centroid for detection in detections]
        
        # If no new detections, mark all existing objects as disappeared
        if len(new_centroids) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].mark_missed()
                
                # Remove objects that have been missed for too long
                if self.objects[obj_id].consecutive_missed > self.max_disappeared:
                    del self.objects[obj_id]
            
            # Return current objects
            return [obj.to_dict() for obj in self.objects.values() if not obj.lost]
        
        # If no existing objects, create new ones
        if len(tracked_centroids) == 0:
            for detection in detections:
                self.objects[f"object-{TrackedObject.id_counter}"] = TrackedObject(detection)
        
        # Calculate distances between each pair of existing and new centroids
        distances = np.zeros((len(tracked_centroids), len(new_centroids)))
        for i, tracked_centroid in enumerate(tracked_centroids):
            for j, new_centroid in enumerate(new_centroids):
                # Calculate Euclidean distance
                d = np.sqrt((tracked_centroid[0] - new_centroid[0])**2 + 
                            (tracked_centroid[1] - new_centroid[1])**2)
                distances[i, j] = d
        
        # Find the smallest distances and associate objects accordingly
        # Using the Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(distances)
        except ImportError:
            # Fallback to greedy assignment if scipy not available
            row_indices = []
            col_indices = []
            
            # Copy the distance matrix to modify
            dist_copy = distances.copy()
            
            # While there are still rows and columns to process
            while dist_copy.size > 0 and dist_copy.shape[0] > 0 and dist_copy.shape[1] > 0:
                # Find the minimum distance
                min_idx = np.unravel_index(np.argmin(dist_copy), dist_copy.shape)
                
                # Add to our matches if the distance is below threshold
                if dist_copy[min_idx] < 100:  # Max distance threshold
                    row_indices.append(min_idx[0])
                    col_indices.append(min_idx[1])
                
                # Remove the row and column
                dist_copy = np.delete(dist_copy, min_idx[0], axis=0)
                dist_copy = np.delete(dist_copy, min_idx[1], axis=1)
        
        # Mark all current objects as not updated yet
        updated_objects = {obj_id: False for obj_id in self.objects}
        
        # Update matched objects
        for row_idx, col_idx in zip(row_indices, col_indices):
            # Only update if the distance is reasonable
            if distances[row_idx, col_idx] < 100:  # Max distance threshold
                # Get the object ID
                obj_id = list(self.objects.keys())[row_idx]
                
                # Update the matched object
                self.objects[obj_id].update(detections[col_idx])
                updated_objects[obj_id] = True
        
        # Add new objects for unmatched detections
        unmatched_cols = set(range(len(new_centroids))) - set(col_indices)
        for col_idx in unmatched_cols:
            # Create a new tracked object
            new_obj = TrackedObject(detections[col_idx])
            self.objects[new_obj.id] = new_obj
        
        # Handle disappeared objects
        # Make a copy of keys to avoid modification during iteration
        object_ids = list(self.objects.keys())
        for obj_id in object_ids:
            if not updated_objects.get(obj_id, False):
                self.objects[obj_id].mark_missed()
                
                # Remove objects that have been missed for too long
                if self.objects[obj_id].consecutive_missed > self.max_disappeared:
                    del self.objects[obj_id]
        
        # Return current objects
        return [obj.to_dict() for obj in self.objects.values() if not obj.lost]
    
    def _update_sort(self, frame: np.ndarray, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Update using SORT tracking method
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of Detection objects
        
        Returns:
            list: Updated list of tracked objects
        """
        # Convert detections to SORT format: [x1, y1, x2, y2, confidence]
        if not detections:
            # No detections, update with empty detections
            if hasattr(self, 'sort_tracker'):
                track_bbs_ids = self.sort_tracker.update(np.empty((0, 5)))
                
                # Clear existing objects that aren't in the current tracks
                current_ids = set()
                
                # Update existing objects and create new ones
                for x1, y1, x2, y2, obj_id in track_bbs_ids:
                    obj_id = int(obj_id)
                    current_ids.add(obj_id)
                
                # Remove objects that are no longer tracked
                self.objects = {k: v for k, v in self.objects.items() if int(k.split('-')[-1]) in current_ids}
                
                # Return current objects
                return [obj.to_dict() for obj in self.objects.values()]
            else:
                return []
        
        detection_array = np.array([[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence] for d in detections])
        
        # Update SORT tracker
        track_bbs_ids = self.sort_tracker.update(detection_array)
        
        # Clear existing objects that aren't in the current tracks
        current_ids = set()
        
        # Update existing objects and create new ones
        for x1, y1, x2, y2, obj_id in track_bbs_ids:
            obj_id = int(obj_id)
            current_ids.add(obj_id)
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            
            # Find corresponding detection for confidence
            confidence = 0.0
            class_id = 0
            class_name = "person"
            
            for det in detections:
                # Use IoU to find the best match
                iou = self._calculate_iou(bbox, det.bbox)
                if iou > 0.5:  # Threshold for matching
                    confidence = det.confidence
                    class_id = det.class_id
                    class_name = det.class_name
                    break
            
            # Create mock detection for update
            detection = Detection(bbox, confidence, class_id, class_name)
            
            # Check if this object is already being tracked
            obj_key = f"object-{obj_id}"
            if obj_key in self.objects:
                # Update existing object
                self.objects[obj_key].update(detection)
            else:
                # Create new tracked object (with SORT ID)
                new_obj = TrackedObject(detection)
                new_obj.id = obj_key
                self.objects[obj_key] = new_obj
        
        # Remove objects that are no longer tracked
        self.objects = {k: v for k, v in self.objects.items() if int(k.split('-')[-1]) in current_ids}
        
        # Return current objects
        return [obj.to_dict() for obj in self.objects.values()]
    
    def _update_deep_sort(self, frame: np.ndarray, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Update using DeepSORT tracking method
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of Detection objects
        
        Returns:
            list: Updated list of tracked objects
        """
        # Convert detections to DeepSORT format: [x1, y1, x2, y2, confidence]
        if not detections:
            return []
        
        xywhs = []
        confidences = []
        class_ids = []
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            x_center = x1 + w / 2
            y_center = y1 + h / 2
            
            xywhs.append([x_center, y_center, w, h])
            confidences.append(det.confidence)
            class_ids.append(det.class_id)
        
        xywhs = torch.Tensor(xywhs)
        confidences = torch.Tensor(confidences)
        
        # Update DeepSORT tracker
        outputs = self.deep_sort_tracker.update(xywhs, confidences, class_ids, frame)
        
        # Clear existing objects that aren't in the current tracks
        current_ids = set()
        
        # List to store updated tracked objects
        updated_objects = []
        
        # Update existing objects and create new ones
        for output in outputs:
            x1, y1, x2, y2, track_id, class_id, confidence = output
            
            current_ids.add(track_id)
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            
            # Find class name from class_id
            class_name = "person"  # Default
            for det in detections:
                if det.class_id == class_id:
                    class_name = det.class_name
                    break
            
            # Create mock detection for update
            detection = Detection(bbox, confidence, class_id, class_name)
            
            # Check if this object is already being tracked
            obj_key = f"object-{track_id}"
            if obj_key in self.objects:
                # Update existing object
                self.objects[obj_key].update(detection)
            else:
                # Create new tracked object (with DeepSORT ID)
                new_obj = TrackedObject(detection)
                new_obj.id = obj_key
                self.objects[obj_key] = new_obj
            
            updated_objects.append(self.objects[obj_key].to_dict())
        
        # Remove objects that are no longer tracked
        self.objects = {k: v for k, v in self.objects.items() if int(k.split('-')[-1]) in current_ids}
        
        return updated_objects
    
    def _update_opencv(self, frame: np.ndarray, detections: List[Detection]) -> List[Dict[str, Any]]:
        """
        Update using OpenCV tracking methods
        
        Args:
            frame (numpy.ndarray): Current frame
            detections (list): List of Detection objects
        
        Returns:
            list: Updated list of tracked objects
        """
        # Create trackers for new detections
        for detection in detections:
            # Check if this detection can be matched to existing object
            matched = False
            for obj_id, obj in list(self.objects.items()):
                # Use IoU to check if this is the same object
                if self._calculate_iou(detection.bbox, obj.bbox) > self.iou_threshold:
                    # Update existing object
                    obj.update(detection)
                    matched = True
                    
                    # Reset the tracker for this object
                    self._create_tracker(obj_id, frame, detection.bbox)
                    break
            
            if not matched:
                # Create new object
                new_obj = TrackedObject(detection)
                self.objects[new_obj.id] = new_obj
                
                # Create a tracker for the new object
                self._create_tracker(new_obj.id, frame, detection.bbox)
        
        # Update objects with trackers
        for obj_id in list(self.objects.keys()):
            if obj_id in self.cv_trackers:
                # Update tracker
                success, bbox = self.cv_trackers[obj_id].update(frame)
                
                if success:
                    # Format as [x1, y1, x2, y2]
                    x, y, w, h = list(map(int, bbox))
                    bbox = [x, y, x + w, y + h]
                    
                    # Update object with tracker result
                    detection = Detection(bbox, self.objects[obj_id].confidence)
                    self.objects[obj_id].update(detection)
                else:
                    # Tracker lost the object
                    self.objects[obj_id].mark_missed()
                    del self.cv_trackers[obj_id]
            else:
                # No tracker for this object
                self.objects[obj_id].mark_missed()
            
            # Remove lost objects
            if self.objects[obj_id].consecutive_missed > self.max_disappeared:
                del self.objects[obj_id]
                if obj_id in self.cv_trackers:
                    del self.cv_trackers[obj_id]
        
        # Return current objects
        return [obj.to_dict() for obj in self.objects.values() if not obj.lost]
    
    def _create_tracker(self, obj_id: str, frame: np.ndarray, bbox: List[int]):
        """
        Create a new OpenCV tracker
        
        Args:
            obj_id (str): Object ID
            frame (numpy.ndarray): Current frame
            bbox (list): Bounding box [x1, y1, x2, y2]
        """
        # Convert bbox format from [x1, y1, x2, y2] to [x, y, w, h]
        x1, y1, x2, y2 = bbox
        cv_bbox = (x1, y1, x2 - x1, y2 - y1)
        
        # Create appropriate tracker
        if self.cv_tracker_type == "KCF":
            tracker = cv2.TrackerKCF_create()
        elif self.cv_tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        elif self.cv_tracker_type == "MOSSE":
            tracker = cv2.legacy.TrackerMOSSE_create()
        elif self.cv_tracker_type == "MIL":
            tracker = cv2.TrackerMIL_create()
        else:
            # Default to KCF
            tracker = cv2.TrackerKCF_create()
        
        # Initialize tracker
        tracker.init(frame, cv_bbox)
        
        # Store tracker
        self.cv_trackers[obj_id] = tracker
    
    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tracks on frame
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            numpy.ndarray: Frame with tracks drawn
        """
        # Make a copy of the frame
        output = frame.copy()
        
        # Draw each tracked object
        for obj_id, obj in self.objects.items():
            if obj.lost:
                continue
            
            # Get bounding box
            x1, y1, x2, y2 = obj.bbox
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID
            cv2.putText(output, obj_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
            
            # Draw path
            if len(obj.path) > 1:
                # Convert to numpy array
                path = np.array(obj.path)
                
                # Draw path line
                for i in range(1, len(path)):
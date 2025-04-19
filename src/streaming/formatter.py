"""
Data formatter for standardizing detection and recognition results
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.logger import setup_logger

logger = setup_logger(__name__)

class DetectionFormatter:
    """Formatter for detection and recognition results"""
    
    def __init__(self):
        """Initialize the formatter"""
        logger.info("Detection formatter initialized")
    
    def format_detection(
        self,
        frame_id: int,
        camera_id: str,
        detections: List[Dict],
        faces: Optional[List[Dict]] = None,
        analytics: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Format detection results
        
        Args:
            frame_id (int): Frame identifier
            camera_id (str): Camera identifier
            detections (list): List of person detections
            faces (list, optional): List of face detections
            analytics (dict, optional): Additional analytics data
        
        Returns:
            dict: Formatted detection data
        """
        try:
            # Base message structure
            formatted_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'frame_id': frame_id,
                'camera_id': camera_id,
                'detections': []
            }
            
            # Format person detections
            for detection in detections:
                formatted_detection = self.format_person_detection(detection)
                if formatted_detection:
                    formatted_data['detections'].append(formatted_detection)
            
            # Add face detections if present
            if faces:
                formatted_data['faces'] = [
                    self.format_face_detection(face)
                    for face in faces
                    if face is not None
                ]
            
            # Add or generate analytics
            formatted_data['analytics'] = self.generate_analytics(
                formatted_data['detections'],
                formatted_data.get('faces', []),
                analytics
            )
            
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error formatting detection data: {e}")
            return self.create_error_message("Error formatting detection data")
    
    def format_person_detection(self, detection: Dict) -> Optional[Dict[str, Any]]:
        """
        Format a person detection
        
        Args:
            detection (dict): Person detection data
        
        Returns:
            dict: Formatted person detection
        """
        try:
            # Ensure required fields
            if 'bbox' not in detection:
                logger.warning("Detection missing required bbox field")
                return None
            
            # Create formatted detection
            formatted = {
                'id': detection.get('id', f"person-{uuid.uuid4().hex[:8]}"),
                'type': 'person',
                'confidence': float(detection.get('confidence', 0.0)),
                'bbox': [int(x) for x in detection['bbox']],
                'recognized': bool(detection.get('recognized', False))
            }
            
            # Add person data if available
            if 'person_data' in detection and detection['person_data']:
                formatted['person_data'] = {
                    'id': detection['person_data'].get('id', ''),
                    'name': detection['person_data'].get('name', ''),
                    'role': detection['person_data'].get('role', ''),
                    'registration_time': detection['person_data'].get(
                        'registration_time',
                        datetime.utcnow().isoformat()
                    )
                }
            
            # Add tracking data if available
            if 'tracking' in detection:
                formatted['tracking'] = {
                    'track_id': detection['tracking'].get('track_id', ''),
                    'velocity': detection['tracking'].get('velocity', [0, 0]),
                    'age': detection['tracking'].get('age', 0)
                }
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting person detection: {e}")
            return None
    
    def format_face_detection(self, face: Dict) -> Optional[Dict[str, Any]]:
        """
        Format a face detection
        
        Args:
            face (dict): Face detection data
        
        Returns:
            dict: Formatted face detection
        """
        try:
            # Ensure required fields
            if 'bbox' not in face:
                logger.warning("Face detection missing required bbox field")
                return None
            
            # Create formatted face detection
            formatted = {
                'id': face.get('id', f"face-{uuid.uuid4().hex[:8]}"),
                'bbox': [int(x) for x in face['bbox']],
                'confidence': float(face.get('confidence', 0.0)),
                'person_id': face.get('person_id', None)
            }
            
            # Add landmarks if available
            if 'landmarks' in face:
                formatted['landmarks'] = {
                    str(k): [float(x) for x in v]
                    for k, v in face['landmarks'].items()
                }
            
            # Add matching data if available
            if 'matching' in face:
                formatted['matching'] = {
                    'matched': bool(face['matching'].get('matched', False)),
                    'similarity': float(face['matching'].get('similarity', 0.0)),
                    'match_id': face['matching'].get('match_id', None)
                }
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting face detection: {e}")
            return None
    
    def generate_analytics(
        self,
        detections: List[Dict],
        faces: List[Dict],
        custom_analytics: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate analytics data
        
        Args:
            detections (list): List of formatted detections
            faces (list): List of formatted faces
            custom_analytics (dict, optional): Custom analytics data
        
        Returns:
            dict: Analytics data
        """
        try:
            # Base analytics
            analytics = {
                'people_count': len(detections),
                'recognized_count': len([
                    d for d in detections
                    if d.get('recognized', False)
                ]),
                'face_count': len(faces),
                'matched_faces': len([
                    f for f in faces
                    if f.get('matching', {}).get('matched', False)
                ])
            }
            
            # Add detection confidence stats
            if detections:
                confidences = [d['confidence'] for d in detections]
                analytics['detection_confidence'] = {
                    'min': min(confidences),
                    'max': max(confidences),
                    'avg': sum(confidences) / len(confidences)
                }
            
            # Add face matching stats
            if faces:
                similarities = [
                    f['matching']['similarity']
                    for f in faces
                    if 'matching' in f and 'similarity' in f['matching']
                ]
                if similarities:
                    analytics['face_matching'] = {
                        'min_similarity': min(similarities),
                        'max_similarity': max(similarities),
                        'avg_similarity': sum(similarities) / len(similarities)
                    }
            
            # Add custom analytics
            if custom_analytics:
                analytics.update(custom_analytics)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating analytics: {e}")
            return {
                'people_count': len(detections),
                'face_count': len(faces),
                'error': str(e)
            }
    
    def create_error_message(self, error: str) -> Dict[str, Any]:
        """
        Create an error message
        
        Args:
            error (str): Error description
        
        Returns:
            dict: Error message
        """
        return {
            'error': error,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def format_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        camera_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format an event message
        
        Args:
            event_type (str): Type of event
            data (dict): Event data
            camera_id (str, optional): Associated camera ID
        
        Returns:
            dict: Formatted event message
        """
        try:
            event = {
                'type': 'event',
                'event_type': event_type,
                'timestamp': datetime.utcnow().isoformat(),
                'data': data
            }
            
            if camera_id:
                event['camera_id'] = camera_id
            
            return event
            
        except Exception as e:
            logger.error(f"Error formatting event: {e}")
            return self.create_error_message("Error formatting event")
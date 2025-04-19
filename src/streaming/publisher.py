"""
Stream publisher implementation for ScaleTent
Handles publishing detection results to subscribers
"""

import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime
from collections import deque

from core.logger import setup_logger

logger = setup_logger(__name__)

class StreamPublisher:
    """Publisher for detection result streams"""
    
    def __init__(self, max_queue_size: int = 100):
        """
        Initialize stream publisher
        
        Args:
            max_queue_size (int): Maximum size of message queue
        """
        self.max_queue_size = max_queue_size
        self.message_queue = asyncio.Queue(maxsize=max_queue_size)
        self.recent_messages = deque(maxlen=max_queue_size)
        
        # Statistics
        self.stats = {
            'messages_published': 0,
            'messages_dropped': 0,
            'start_time': datetime.utcnow()
        }
        
        logger.info(f"Stream publisher initialized with queue size {max_queue_size}")
    
    async def publish(self, message: Dict[str, Any]) -> bool:
        """
        Publish a message to the stream
        
        Args:
            message (dict): Message to publish
        
        Returns:
            bool: True if published successfully, False if dropped
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in message:
                message['timestamp'] = datetime.utcnow().isoformat()
            
            # Try to put message in queue
            try:
                await asyncio.wait_for(
                    self.message_queue.put(message),
                    timeout=0.1  # 100ms timeout
                )
                
                # Store in recent messages
                self.recent_messages.append(message)
                
                self.stats['messages_published'] += 1
                return True
                
            except asyncio.TimeoutError:
                logger.warning("Message queue full, dropping message")
                self.stats['messages_dropped'] += 1
                return False
            
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False
    
    async def get_message(self) -> Optional[Dict[str, Any]]:
        """
        Get next message from queue
        
        Returns:
            dict: Next message or None if queue is empty
        """
        try:
            # Get message with timeout
            message = await asyncio.wait_for(
                self.message_queue.get(),
                timeout=0.1  # 100ms timeout
            )
            
            # Mark task as done
            self.message_queue.task_done()
            
            return message
            
        except asyncio.TimeoutError:
            return None
            
        except Exception as e:
            logger.error(f"Error getting message: {e}")
            return None
    
    def get_recent_messages(self, count: int = None) -> list:
        """
        Get recent messages
        
        Args:
            count (int, optional): Number of messages to return
        
        Returns:
            list: List of recent messages
        """
        if count is None:
            return list(self.recent_messages)
        return list(self.recent_messages)[-count:]
    
    async def publish_detection(
        self,
        frame_id: int,
        camera_id: str,
        detections: list,
        faces: list = None,
        analytics: Dict = None
    ) -> bool:
        """
        Publish detection results
        
        Args:
            frame_id (int): Frame identifier
            camera_id (str): Camera identifier
            detections (list): List of person detections
            faces (list, optional): List of face detections
            analytics (dict, optional): Additional analytics data
        
        Returns:
            bool: True if published successfully
        """
        try:
            message = {
                'frame_id': frame_id,
                'camera_id': camera_id,
                'timestamp': datetime.utcnow().isoformat(),
                'detections': detections
            }
            
            if faces:
                message['faces'] = faces
            
            if analytics:
                message['analytics'] = analytics
            
            return await self.publish(message)
            
        except Exception as e:
            logger.error(f"Error publishing detection: {e}")
            return False
    
    async def publish_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        camera_id: str = None
    ) -> bool:
        """
        Publish an event
        
        Args:
            event_type (str): Type of event
            data (dict): Event data
            camera_id (str, optional): Associated camera ID
        
        Returns:
            bool: True if published successfully
        """
        try:
            message = {
                'type': 'event',
                'event_type': event_type,
                'timestamp': datetime.utcnow().isoformat(),
                'data': data
            }
            
            if camera_id:
                message['camera_id'] = camera_id
            
            return await self.publish(message)
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics"""
        return {
            'messages_published': self.stats['messages_published'],
            'messages_dropped': self.stats['messages_dropped'],
            'queue_size': self.message_queue.qsize(),
            'queue_full': self.message_queue.full(),
            'uptime_seconds': (
                datetime.utcnow() - self.stats['start_time']
            ).total_seconds()
        }
    
    async def wait_empty(self, timeout: float = None):
        """
        Wait for message queue to be empty
        
        Args:
            timeout (float, optional): Maximum time to wait in seconds
        """
        try:
            await asyncio.wait_for(
                self.message_queue.join(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for message queue to empty")
        except Exception as e:
            logger.error(f"Error waiting for queue: {e}")
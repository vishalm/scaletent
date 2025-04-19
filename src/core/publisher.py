"""
Publisher module for real-time updates and message broadcasting
"""

import asyncio
import json
import logging
from collections import deque
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.core.logger import setup_logger

logger = setup_logger(__name__)

class Publisher:
    """
    Publisher class for managing real-time updates and message broadcasting.
    Maintains a queue of recent messages and handles message distribution to subscribers.
    """

    def __init__(self, max_recent_messages: int = 100):
        """
        Initialize the publisher.
        
        Args:
            max_recent_messages (int): Maximum number of recent messages to keep in memory
        """
        self.subscribers = set()
        self.recent_messages = deque(maxlen=max_recent_messages)
        self.lock = asyncio.Lock()
        logger.info("Publisher initialized")

    async def subscribe(self, subscriber):
        """
        Add a new subscriber to receive messages.
        
        Args:
            subscriber: WebSocket connection to subscribe
        """
        async with self.lock:
            self.subscribers.add(subscriber)
            logger.debug(f"New subscriber added. Total subscribers: {len(self.subscribers)}")

    async def unsubscribe(self, subscriber):
        """
        Remove a subscriber from the message distribution list.
        
        Args:
            subscriber: WebSocket connection to unsubscribe
        """
        async with self.lock:
            self.subscribers.remove(subscriber)
            logger.debug(f"Subscriber removed. Total subscribers: {len(self.subscribers)}")

    async def publish(self, message: Dict[str, Any], topic: Optional[str] = None):
        """
        Publish a message to all subscribers.
        
        Args:
            message (Dict[str, Any]): Message to publish
            topic (Optional[str]): Topic to publish the message under
        """
        try:
            # Add timestamp and topic if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.utcnow().isoformat()
            if topic and "topic" not in message:
                message["topic"] = topic

            # Convert message to JSON string
            message_str = json.dumps(message)
            
            # Store in recent messages
            self.recent_messages.append(message_str)

            # Send to all subscribers
            async with self.lock:
                disconnected = set()
                for subscriber in self.subscribers:
                    try:
                        await subscriber.send_text(message_str)
                    except Exception as e:
                        logger.warning(f"Failed to send message to subscriber: {e}")
                        disconnected.add(subscriber)

                # Remove disconnected subscribers
                for subscriber in disconnected:
                    self.subscribers.remove(subscriber)

        except Exception as e:
            logger.error(f"Error publishing message: {e}")

    def get_recent_messages(self, limit: Optional[int] = None) -> List[str]:
        """
        Get recent messages from the queue.
        
        Args:
            limit (Optional[int]): Maximum number of messages to return
            
        Returns:
            List[str]: List of recent messages as JSON strings
        """
        if limit is None or limit >= len(self.recent_messages):
            return list(self.recent_messages)
        return list(self.recent_messages)[-limit:]

    @property
    def subscriber_count(self) -> int:
        """Get the current number of subscribers."""
        return len(self.subscribers)

    def clear_messages(self):
        """Clear all stored messages."""
        self.recent_messages.clear() 
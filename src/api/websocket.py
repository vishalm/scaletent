"""
WebSocket server implementation for real-time data streaming
"""

import asyncio
import json
import websockets
from typing import Set, Dict, Any
from datetime import datetime

from core.logger import setup_logger

logger = setup_logger(__name__)

class WebSocketServer:
    """WebSocket server for streaming detection results"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8765, publisher=None):
        """
        Initialize WebSocket server
        
        Args:
            host (str): Host address to bind to
            port (int): Port to listen on
            publisher: Stream publisher instance
        """
        self.host = host
        self.port = port
        self.publisher = publisher
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.server = None
        self.running = False
        
        # Statistics
        self.stats = {
            'total_connections': 0,
            'messages_sent': 0,
            'start_time': None
        }
        
        logger.info(f"WebSocket server initialized on {host}:{port}")
    
    async def start(self):
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.handle_client,
                self.host,
                self.port
            )
            
            self.running = True
            self.stats['start_time'] = datetime.utcnow()
            
            logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
            
            # Start message processing
            if self.publisher:
                asyncio.create_task(self.process_messages())
        
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def stop(self):
        """Stop the WebSocket server"""
        try:
            logger.info("Stopping WebSocket server")
            
            # Close all client connections
            if self.clients:
                await asyncio.gather(*[
                    client.close() for client in self.clients
                ])
            
            # Stop server
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            
            self.running = False
            logger.info("WebSocket server stopped")
        
        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {e}")
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """
        Handle new client connection
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        try:
            # Register client
            self.clients.add(websocket)
            self.stats['total_connections'] += 1
            
            client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            logger.info(f"New client connected: {client_info}")
            
            try:
                # Keep connection alive and handle messages
                async for message in websocket:
                    try:
                        # Parse client message
                        data = json.loads(message)
                        
                        # Handle client commands
                        if 'command' in data:
                            await self.handle_command(websocket, data)
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from client: {client_info}")
                    
                    except Exception as e:
                        logger.error(f"Error handling client message: {e}")
            
            finally:
                # Unregister client on disconnect
                self.clients.remove(websocket)
                logger.info(f"Client disconnected: {client_info}")
        
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
    
    async def handle_command(self, websocket: websockets.WebSocketServerProtocol, data: Dict):
        """
        Handle client commands
        
        Args:
            websocket: Client WebSocket connection
            data: Command data
        """
        command = data.get('command')
        
        if command == 'get_stats':
            # Send server statistics
            await self.send_stats(websocket)
        
        elif command == 'subscribe':
            # Handle stream subscription
            streams = data.get('streams', [])
            if streams:
                logger.info(f"Client subscribed to streams: {streams}")
        
        else:
            logger.warning(f"Unknown command: {command}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast message to all connected clients
        
        Args:
            message (dict): Message to broadcast
        """
        if not self.clients:
            return
        
        try:
            # Prepare message
            message_str = json.dumps(message)
            
            # Send to all clients
            await asyncio.gather(*[
                client.send(message_str)
                for client in self.clients
            ])
            
            self.stats['messages_sent'] += len(self.clients)
        
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
    
    async def process_messages(self):
        """Process and broadcast messages from publisher"""
        if not self.publisher:
            logger.warning("No publisher configured for WebSocket server")
            return
        
        try:
            while self.running:
                # Get next message from publisher
                message = await self.publisher.get_message()
                
                if message:
                    # Add timestamp
                    message['timestamp'] = datetime.utcnow().isoformat()
                    
                    # Broadcast to clients
                    await self.broadcast(message)
                
                # Small delay to prevent busy loop
                await asyncio.sleep(0.001)
        
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
    
    async def send_stats(self, websocket: websockets.WebSocketServerProtocol):
        """
        Send server statistics to client
        
        Args:
            websocket: Client WebSocket connection
        """
        try:
            stats = {
                'total_connections': self.stats['total_connections'],
                'current_connections': len(self.clients),
                'messages_sent': self.stats['messages_sent'],
                'uptime_seconds': (
                    datetime.utcnow() - self.stats['start_time']
                ).total_seconds() if self.stats['start_time'] else 0
            }
            
            await websocket.send(json.dumps({
                'type': 'stats',
                'data': stats
            }))
        
        except Exception as e:
            logger.error(f"Error sending stats: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            'total_connections': self.stats['total_connections'],
            'current_connections': len(self.clients),
            'messages_sent': self.stats['messages_sent'],
            'uptime_seconds': (
                datetime.utcnow() - self.stats['start_time']
            ).total_seconds() if self.stats['start_time'] else 0
        }
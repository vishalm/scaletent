import pytest
import asyncio
import json
import websockets
import numpy as np
import torch
from unittest.mock import Mock, patch
import platform

# Skip tests if not on Apple Silicon
requires_apple_silicon = pytest.mark.skipif(
    platform.processor() != "arm",
    reason="Test requires Apple Silicon"
)

@pytest.fixture
def mock_stream_data():
    """Create mock streaming data."""
    return {
        "timestamp": "2025-04-19T14:30:25.123Z",
        "frame_id": 4562,
        "camera_id": "main-entrance-01",
        "detections": [
            {
                "id": "person-1",
                "type": "person",
                "confidence": 0.97,
                "bbox": [120, 80, 210, 380],
                "recognized": True,
                "person_data": {
                    "name": "John Smith",
                    "id": "JS001",
                    "role": "Attendee"
                }
            }
        ]
    }

@pytest.fixture
def mock_websocket_server():
    """Create a mock WebSocket server."""
    async def echo(websocket, path):
        async for message in websocket:
            await websocket.send(message)
    
    return echo

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection and basic communication."""
    async def handler(websocket, path):
        await websocket.send(json.dumps({"status": "connected"}))
    
    server = await websockets.serve(handler, "localhost", 8765)
    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            response = await websocket.recv()
            data = json.loads(response)
            assert data["status"] == "connected"
    finally:
        server.close()
        await server.wait_closed()

@pytest.mark.asyncio
async def test_stream_data_format(mock_stream_data):
    """Test streaming data format and validation."""
    # Validate required fields
    assert "timestamp" in mock_stream_data
    assert "frame_id" in mock_stream_data
    assert "camera_id" in mock_stream_data
    assert "detections" in mock_stream_data
    
    # Validate detection format
    detection = mock_stream_data["detections"][0]
    assert "id" in detection
    assert "confidence" in detection
    assert "bbox" in detection
    assert len(detection["bbox"]) == 4

@requires_apple_silicon
@pytest.mark.asyncio
async def test_video_stream_processing():
    """Test video stream processing on Apple Silicon."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Mock video frame
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(device)
    
    # Process frame
    try:
        # Simulate frame processing
        processed = torch.nn.functional.interpolate(
            frame_tensor.unsqueeze(0),
            size=(640, 640),
            mode='bilinear',
            align_corners=False
        )
        assert processed.device.type == device.type
        assert processed.shape == (1, 3, 640, 640)
    except Exception as e:
        pytest.fail(f"Video processing failed: {e}")

@pytest.mark.asyncio
async def test_stream_performance():
    """Test streaming performance metrics."""
    import time
    
    # Simulate stream of frames
    num_frames = 30
    frame_times = []
    
    for _ in range(num_frames):
        start_time = time.time()
        
        # Simulate frame processing
        await asyncio.sleep(0.01)  # Simulate processing time
        
        frame_times.append(time.time() - start_time)
    
    # Calculate FPS
    avg_frame_time = sum(frame_times) / len(frame_times)
    fps = 1 / avg_frame_time
    
    assert fps > 15, f"Stream performance below threshold: {fps:.2f} FPS"

@pytest.mark.asyncio
async def test_multiple_clients():
    """Test handling multiple client connections."""
    connected_clients = set()
    
    async def handler(websocket, path):
        connected_clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            connected_clients.remove(websocket)
    
    server = await websockets.serve(handler, "localhost", 8765)
    try:
        # Connect multiple clients
        clients = []
        for _ in range(3):
            client = await websockets.connect("ws://localhost:8765")
            clients.append(client)
        
        assert len(connected_clients) == 3, "Failed to handle multiple clients"
        
        # Cleanup
        for client in clients:
            await client.close()
    finally:
        server.close()
        await server.wait_closed()

@pytest.mark.asyncio
async def test_error_handling_stream():
    """Test error handling in streaming pipeline."""
    # Test invalid WebSocket URL
    with pytest.raises(websockets.exceptions.InvalidURI):
        async with websockets.connect("invalid_url"):
            pass
    
    # Test connection timeout
    with pytest.raises(asyncio.TimeoutError):
        async with websockets.connect("ws://non-existent-server:8765", timeout=1):
            pass

@pytest.mark.asyncio
async def test_stream_reconnection():
    """Test stream reconnection logic."""
    connection_attempts = 0
    max_attempts = 3
    
    async def attempt_connection():
        nonlocal connection_attempts
        connection_attempts += 1
        if connection_attempts < max_attempts:
            raise websockets.exceptions.ConnectionClosed(1006, "Connection lost")
        return True
    
    # Simulate reconnection attempts
    try:
        result = await attempt_connection()
        while not result and connection_attempts < max_attempts:
            await asyncio.sleep(0.1)
            result = await attempt_connection()
        
        assert result, "Failed to reconnect after multiple attempts"
    except Exception as e:
        pytest.fail(f"Reconnection test failed: {e}")

@pytest.mark.asyncio
async def test_stream_data_validation(mock_stream_data):
    """Test streaming data validation."""
    # Test timestamp format
    from datetime import datetime
    try:
        datetime.strptime(mock_stream_data["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        pytest.fail("Invalid timestamp format")
    
    # Test numeric fields
    assert isinstance(mock_stream_data["frame_id"], int)
    detection = mock_stream_data["detections"][0]
    assert isinstance(detection["confidence"], float)
    assert all(isinstance(x, int) for x in detection["bbox"])

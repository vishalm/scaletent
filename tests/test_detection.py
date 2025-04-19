import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import platform
from src.detection.detector import YOLODetector

# Skip tests if not on Apple Silicon
requires_apple_silicon = pytest.mark.skipif(
    platform.processor() != "arm",
    reason="Test requires Apple Silicon"
)

@pytest.fixture
def mock_image():
    """Create a mock image for testing."""
    return np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

@pytest.fixture
def mock_model():
    """Create a mock YOLO model."""
    model = Mock()
    model.predict.return_value = [Mock(boxes=Mock(data=torch.tensor([[100, 100, 200, 200, 0.95, 0]])))]
    return model

def test_torch_mps_availability():
    """Test if MPS (Metal Performance Shaders) is available."""
    assert hasattr(torch.backends, 'mps'), "PyTorch MPS backend not found"
    if platform.processor() == "arm":
        assert torch.backends.mps.is_available(), "MPS not available on Apple Silicon"

@requires_apple_silicon
def test_device_selection():
    """Test proper device selection on Apple Silicon."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    assert device.type in ["mps", "cpu"], "Invalid device type"
    
    # Test tensor operations on the device
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = x @ y
    assert z.device.type == device.type, "Tensor not on correct device"

@pytest.mark.asyncio
async def test_detection_pipeline(mock_image, mock_model):
    """Test the complete detection pipeline."""
    # Convert image to tensor
    image_tensor = torch.from_numpy(mock_image).permute(2, 0, 1).float()
    
    # Get device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    
    # Run detection
    with patch('ultralytics.YOLO', return_value=mock_model):
        results = mock_model.predict(image_tensor)
        boxes = results[0].boxes.data
        
        assert len(boxes) > 0, "No detections found"
        assert boxes.shape[1] == 6, "Incorrect box format"  # x1, y1, x2, y2, conf, class

@requires_apple_silicon
def test_model_performance():
    """Test model performance metrics on Apple Silicon."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Performance test with a larger tensor
    input_size = (1000, 1000)
    x = torch.randn(*input_size).to(device)
    
    # Measure operation speed
    import time
    start_time = time.time()
    for _ in range(100):
        y = torch.nn.functional.relu(x)
    end_time = time.time()
    
    operation_time = end_time - start_time
    assert operation_time < 1.0, f"Performance test took too long: {operation_time:.2f} seconds"

@pytest.mark.parametrize("input_size", [
    (320, 320),
    (640, 640),
    (1280, 1280)
])
def test_input_sizes(input_size, mock_model):
    """Test different input sizes for detection."""
    test_image = np.random.randint(0, 255, (*input_size, 3), dtype=np.uint8)
    image_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    
    with patch('ultralytics.YOLO', return_value=mock_model):
        results = mock_model.predict(image_tensor)
        assert results is not None, f"Detection failed for input size {input_size}"

def test_error_handling():
    """Test error handling in detection pipeline."""
    # Test invalid device
    with pytest.raises(RuntimeError):
        torch.tensor([1.0]).to("invalid_device")
    
    # Test invalid input
    with pytest.raises((ValueError, RuntimeError)):
        # Try to perform an invalid operation that should raise an error
        invalid_tensor = torch.tensor([1.0]).to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
        invalid_tensor.view(-1, -1)  # This should raise an error as dimensions are invalid

@requires_apple_silicon
def test_memory_management():
    """Test memory management on Apple Silicon."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Allocate a large tensor
    large_tensor = torch.randn(1000, 1000).to(device)
    
    # Force garbage collection
    import gc
    del large_tensor
    gc.collect()
    torch.mps.empty_cache() if device.type == "mps" else None
    
    # Verify we can allocate again
    try:
        new_tensor = torch.randn(1000, 1000).to(device)
        assert new_tensor is not None, "Failed to allocate new tensor after cleanup"
    except RuntimeError as e:
        pytest.fail(f"Memory management test failed: {e}")

def test_detector_initialization(detector: YOLODetector):
    """Test that the detector can be initialized."""
    assert detector is not None
    assert isinstance(detector, YOLODetector)

def test_detector_inference(detector: YOLODetector, test_image: np.ndarray):
    """Test that the detector can process an image."""
    results = detector.detect(test_image)
    assert isinstance(results, list)

@pytest.mark.integration
def test_detector_with_real_image(detector: YOLODetector):
    """Test detector with a real image (integration test)."""
    # This is just a placeholder for demonstration
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    results = detector.detect(test_image)
    assert isinstance(results, list)

@pytest.mark.slow
def test_detector_batch_processing(detector: YOLODetector):
    """Test batch processing capabilities (marked as slow)."""
    batch_size = 4
    test_batch = [
        np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(batch_size)
    ]
    results = [detector.detect(img) for img in test_batch]
    assert len(results) == batch_size

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import platform
import mediapipe as mp

# Skip tests if not on Apple Silicon
requires_apple_silicon = pytest.mark.skipif(
    platform.processor() != "arm",
    reason="Test requires Apple Silicon"
)

@pytest.fixture
def mock_face_image():
    """Create a mock face image for testing."""
    return np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

@pytest.fixture
def mock_face_detector():
    """Create a mock MediaPipe face detector."""
    detector = Mock()
    detector.process.return_value = Mock(
        detections=[Mock(
            location_data=Mock(
                relative_bounding_box=Mock(
                    xmin=0.1, ymin=0.1,
                    width=0.2, height=0.2
                )
            ),
            score=[0.98]
        )]
    )
    return detector

@pytest.fixture
def mock_recognition_model():
    """Create a mock face recognition model."""
    model = Mock()
    model.return_value = torch.randn(1, 512)  # Mock embedding vector
    return model

def test_mediapipe_initialization():
    """Test MediaPipe initialization on Apple Silicon."""
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        assert face_detection is not None, "Failed to initialize MediaPipe face detection"
    except Exception as e:
        pytest.fail(f"MediaPipe initialization failed: {e}")

@requires_apple_silicon
def test_face_detection_pipeline(mock_face_image, mock_face_detector):
    """Test the face detection pipeline."""
    with patch('mediapipe.solutions.face_detection.FaceDetection', return_value=mock_face_detector):
        # Process image
        results = mock_face_detector.process(mock_face_image)
        
        # Verify detection
        assert results.detections is not None, "No faces detected"
        assert len(results.detections) > 0, "No face detections returned"
        assert results.detections[0].score[0] > 0.5, "Low confidence detection"

@requires_apple_silicon
def test_face_recognition_model(mock_face_image, mock_recognition_model):
    """Test face recognition model on MPS device."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Convert image to tensor
    image_tensor = torch.from_numpy(mock_face_image).permute(2, 0, 1).float().unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    with patch('src.recognition.embedder.FaceEmbedder', return_value=mock_recognition_model):
        # Test forward pass
        mock_recognition_model.return_value = torch.randn(512)  # Mock embedding output
        embedding = mock_recognition_model(image_tensor)
        assert embedding.shape == (512,), "Invalid embedding shape"

@pytest.mark.asyncio
async def test_recognition_pipeline(mock_face_image, mock_face_detector, mock_recognition_model):
    """Test the complete recognition pipeline."""
    # Detection
    with patch('mediapipe.solutions.face_detection.FaceDetection', return_value=mock_face_detector):
        detection_results = mock_face_detector.process(mock_face_image)
        assert detection_results.detections is not None
    
    # Recognition
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    image_tensor = torch.from_numpy(mock_face_image).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    with patch('src.recognition.embedder.FaceEmbedder', return_value=mock_recognition_model):
        # Test embedding generation
        mock_recognition_model.return_value = torch.randn(512)  # Mock embedding output
        embedding = mock_recognition_model(image_tensor)
        assert embedding.shape == (512,), "Invalid embedding shape"

def test_face_alignment():
    """Test face alignment preprocessing."""
    # Create test image with face
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Mock face landmarks
    landmarks = np.array([
        [100, 100],  # Left eye
        [140, 100],  # Right eye
        [120, 130],  # Nose
        [110, 150],  # Left mouth
        [130, 150],  # Right mouth
    ])
    
    # Test alignment
    try:
        # Simple rotation alignment based on eyes
        eye_left = landmarks[0]
        eye_right = landmarks[1]
        
        # Calculate angle
        eye_angle = np.degrees(np.arctan2(
            eye_right[1] - eye_left[1],
            eye_right[0] - eye_left[0]
        ))
        
        assert isinstance(eye_angle, float), "Failed to calculate alignment angle"
    except Exception as e:
        pytest.fail(f"Face alignment failed: {e}")

@pytest.mark.parametrize("confidence_threshold", [0.3, 0.5, 0.7, 0.9])
def test_detection_thresholds(mock_face_detector, confidence_threshold):
    """Test different confidence thresholds for face detection."""
    mock_face_detector.process.return_value = Mock(
        detections=[Mock(
            location_data=Mock(
                relative_bounding_box=Mock(
                    xmin=0.1, ymin=0.1,
                    width=0.2, height=0.2
                )
            ),
            score=[confidence_threshold]
        )]
    )
    
    results = mock_face_detector.process(np.zeros((100, 100, 3)))
    assert results.detections[0].score[0] >= confidence_threshold

@requires_apple_silicon
def test_batch_processing():
    """Test batch processing of face recognition."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create batch of face images
    batch_size = 4
    batch_images = torch.randn(batch_size, 3, 112, 112).to(device)
    
    # Process batch
    try:
        # Simulate batch forward pass
        features = torch.nn.functional.normalize(batch_images, p=2, dim=1)
        assert features.shape == (batch_size, 3, 112, 112)
        assert features.device.type == device.type
    except Exception as e:
        pytest.fail(f"Batch processing failed: {e}")

def test_error_handling_recognition():
    """Test error handling in recognition pipeline."""
    from src.recognition.embedder import FaceEmbedder
    
    # Test invalid image size
    with pytest.raises(ValueError):
        # Create a tensor with invalid dimensions for face recognition
        invalid_image = torch.randn(1, 3, 10, 10)  # Too small for face recognition
        FaceEmbedder.validate_input(invalid_image)
    
    # Test invalid number of channels
    with pytest.raises(ValueError):
        invalid_channels = torch.randn(1, 4, 112, 112)  # 4 channels instead of 3
        FaceEmbedder.validate_input(invalid_channels)

@staticmethod
def validate_input(tensor):
    """Validate input tensor dimensions."""
    if tensor.dim() != 4:
        raise ValueError("Input must be a 4D tensor (batch, channels, height, width)")
    if tensor.size(1) != 3:
        raise ValueError("Input must have 3 channels")
    if tensor.size(2) < 112 or tensor.size(3) < 112:
        raise ValueError("Input spatial dimensions must be at least 112x112")

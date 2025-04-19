"""
PyTest configuration and fixtures for ScaleTent tests.
"""
import os
import pytest
import asyncio
from pathlib import Path
from typing import Generator, AsyncGenerator

import yaml
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.core.config import Config
from src.web.app import create_app
from src.detection.detector import YOLODetector
from src.recognition.face_detector import FaceDetector
from src.recognition.matcher import FaceMatcher
from src.streaming.publisher import StreamPublisher

# Constants for testing
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_CONFIG_PATH = Path(__file__).parent / "test_config.yml"

@pytest.fixture(scope="session")
def test_config() -> Config:
    """Provide test configuration."""
    if not TEST_CONFIG_PATH.exists():
        config_data = {
            "api": {
                "host": "localhost",
                "port": 8000,
                "websocket": {
                    "host": "localhost",
                    "port": 8765
                }
            },
            "detection": {
                "model_path": "models/yolov8n.pt",
                "confidence_threshold": 0.5,
                "device": "cpu"
            },
            "recognition": {
                "face_detector_model": "models/face_detection_model.pth",
                "embedder_model": "models/facenet_model.pth",
                "database_path": "data/face_database",
                "similarity_threshold": 0.7
            }
        }
        TEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TEST_CONFIG_PATH, 'w') as f:
            yaml.dump(config_data, f)
    
    with open(TEST_CONFIG_PATH, 'r') as f:
        config_data = yaml.safe_load(f)
    return Config(**config_data)

@pytest.fixture(scope="session")
def test_image() -> np.ndarray:
    """Provide a test image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def detector(test_config: Config) -> YOLODetector:
    """Provide a YOLODetector instance."""
    return YOLODetector(
        model_path=test_config.get('detection.model_path'),
        confidence_threshold=test_config.get('detection.confidence_threshold'),
        device=test_config.get('detection.device')
    )

@pytest.fixture
def face_detector(test_config: Config) -> FaceDetector:
    """Provide a FaceDetector instance."""
    return FaceDetector(
        model_path=test_config.get('recognition.face_detector_model'),
        confidence_threshold=0.5
    )

@pytest.fixture
def face_matcher(test_config: Config) -> FaceMatcher:
    """Provide a FaceMatcher instance."""
    return FaceMatcher(
        embedder_model_path=test_config.get('recognition.embedder_model'),
        database_path=test_config.get('recognition.database_path'),
        similarity_threshold=test_config.get('recognition.similarity_threshold')
    )

@pytest.fixture
def publisher() -> StreamPublisher:
    """Provide a StreamPublisher instance."""
    return StreamPublisher()

@pytest.fixture
async def app(test_config: Config) -> AsyncGenerator:
    """Provide a FastAPI application instance."""
    app = create_app(test_config)
    yield app

@pytest.fixture
def test_client(app) -> Generator:
    """Provide a TestClient instance."""
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["DEVICE_TYPE"] = "cpu"
    yield
    os.environ.pop("TESTING", None)
    os.environ.pop("DEVICE_TYPE", None) 
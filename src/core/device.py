"""
Device configuration module for ScaleTent
"""

import os
import platform
import torch
from src.core.logger import setup_logger

logger = setup_logger(__name__)

def get_device() -> torch.device:
    """
    Get the appropriate device for PyTorch operations.
    Handles device selection based on system capabilities:
    - CUDA if available
    - MPS if on Apple Silicon
    - CPU as fallback
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device")
    elif platform.processor() == "arm" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device

def get_device_from_config(device_str: str) -> torch.device:
    """
    Get device based on configuration string.
    Falls back to automatic device selection if specified device is not available.
    
    Args:
        device_str: Device string from config ('cuda', 'mps', 'cpu')
    """
    if device_str == "auto":
        return get_device()
    
    if device_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(device_str)
        logger.info(f"Using configured CUDA device: {device_str}")
    elif device_str == "mps" and platform.processor() == "arm" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using configured MPS device")
    else:
        if device_str != "cpu":
            logger.warning(f"Requested device '{device_str}' not available, falling back to CPU")
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device 
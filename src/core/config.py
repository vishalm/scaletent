"""
Configuration management for ScaleTent
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging
from src.core.logger import setup_logger

logger = setup_logger(__name__)

class Config:
    """
    Configuration management class
    """
    
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """
        Initialize configuration with provided values
        """
        self._config = kwargs
        self.config_path = None
        
        # Load configuration
        if kwargs.get('config_path'):
            self.load_from_file(kwargs['config_path'])
        
        self.debug_mode = False
        
        logger.info("Configuration initialized")
    
    def load_from_file(self, config_path):
        """
        Load configuration from file
        
        Args:
            config_path (str): Path to configuration file
        
        Returns:
            bool: Success status
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                logger.error(f"Configuration file not found: {config_path}")
                return False
            
            # Load configuration based on file extension
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
            
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    self._config = json.load(f)
            
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return False
            
            self.config_path = config_path
            logger.info(f"Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def save_to_file(self, config_path=None):
        """
        Save configuration to file
        
        Args:
            config_path (str, optional): Path to save configuration file
        
        Returns:
            bool: Success status
        """
        try:
            config_path = Path(config_path) if config_path else self.config_path
            
            if not config_path:
                logger.error("No configuration path specified")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(config_path.parent, exist_ok=True)
            
            # Save configuration based on file extension
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(self._config, f, default_flow_style=False)
            
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return False
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get configuration value by key
        
        Args:
            key (str): Configuration key (dot-separated for nested values)
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        try:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting configuration value for key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key
        
        Args:
            key (str): Configuration key (dot-separated for nested values)
            value: Value to set
        """
        try:
            keys = key.split('.')
            config = self._config
            
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            
            logger.debug(f"Configuration value set: {key} = {value}")
        except Exception as e:
            logger.error(f"Error setting configuration value for key '{key}': {e}")
    
    def set_debug_mode(self, debug_mode):
        """
        Set debug mode
        
        Args:
            debug_mode (bool): Debug mode enabled/disabled
        """
        self.debug_mode = debug_mode
        logger.info(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
    
    def is_debug_mode(self):
        """
        Check if debug mode is enabled
        
        Returns:
            bool: Debug mode status
        """
        return self.debug_mode
    
    def get_all(self):
        """
        Get all configuration data
        
        Returns:
            dict: All configuration data
        """
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value using dictionary syntax
        
        Args:
            key (str): Configuration key (dot-separated for nested values)
        
        Returns:
            Configuration value
        """
        return self.get(key)
"""
Configuration management for ScaleTent
"""

import os
import yaml
import json
from pathlib import Path

from core.logger import setup_logger

logger = setup_logger(__name__)

class Config:
    """
    Configuration management class
    """
    
    def __init__(self, config_data=None, config_path=None):
        """
        Initialize configuration
        
        Args:
            config_data (dict, optional): Configuration data
            config_path (str, optional): Path to configuration file
        """
        self.config_data = {}
        self.config_path = None
        
        # Load configuration
        if config_path:
            self.load_from_file(config_path)
        
        if config_data:
            self.config_data = config_data
            
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
                    self.config_data = yaml.safe_load(f)
            
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    self.config_data = json.load(f)
            
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
                    yaml.dump(self.config_data, f, default_flow_style=False)
            
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config_data, f, indent=2)
            
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return False
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, key, default=None):
        """
        Get configuration value by key
        
        Args:
            key (str): Configuration key (dot-separated for nested values)
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        try:
            # Split key by dots for nested access
            parts = key.split('.')
            value = self.config_data
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting configuration value for key '{key}': {e}")
            return default
    
    def set(self, key, value):
        """
        Set configuration value by key
        
        Args:
            key (str): Configuration key (dot-separated for nested values)
            value: Value to set
        
        Returns:
            bool: Success status
        """
        try:
            # Split key by dots for nested access
            parts = key.split('.')
            config = self.config_data
            
            # Navigate to the last parent
            for part in parts[:-1]:
                if part not in config:
                    config[part] = {}
                config = config[part]
            
            # Set the value
            config[parts[-1]] = value
            
            logger.debug(f"Configuration value set: {key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration value for key '{key}': {e}")
            return False
    
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
        return self.config_data.copy()
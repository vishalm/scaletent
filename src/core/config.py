"""
Configuration management using SQLite with advanced monitoring
"""

import os
import yaml
from typing import Any, Dict, List, Optional
from src.core.logger import setup_logger
from src.core.config_db import ConfigDB
from pathlib import Path

logger = setup_logger(__name__)

class Config:
    """Configuration management class with advanced monitoring"""
    
    DEFAULT_CONFIG = {
        'general': {
            'debug': False,
            'log_level': 'INFO',
            'data_dir': 'data',
            'metrics_enabled': True,
            'auto_recovery': True,
            'health_check_interval': 60  # seconds
        },
        'detection': {
            'min_face_size': 20,
            'confidence_threshold': 0.5,
            'track_metrics': True,
            'save_debug_frames': False,
            'max_faces': 10,
            'performance_mode': 'balanced',
            'batch_size': 4
        },
        'recognition': {
            'similarity_threshold': 0.6,
            'max_distance': 1.0,
            'track_metrics': True,
            'save_embeddings': True,
            'cache_embeddings': True,
            'cache_size': 1000,
            'batch_size': 4
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 1,
            'enable_docs': True,
            'cors_origins': ['*'],
            'rate_limit': 100,  # requests per minute
            'debug': False,
            'api_key': None
        },
        'metrics': {
            'enabled': True,
            'save_interval': 300,  # 5 minutes
            'retention_days': 7,
            'detailed_logging': True,
            'export_format': 'csv',
            'alert_thresholds': {
                'high_cpu_usage': 80,  # percentage
                'high_memory_usage': 80,  # percentage
                'low_fps': 10,
                'high_latency': 100  # milliseconds
            }
        },
        'storage': {
            'max_frame_age': 7,  # days
            'max_storage_size': 10,  # GB
            'compression': True,
            'backup_enabled': True,
            'backup_interval': 86400  # 24 hours
        }
    }
    
    DEFAULT_CAMERA = {
        'id': 'mac-camera',
        'name': 'Mac Built-in Camera',
        'source': '0',  # Use 0 for built-in camera
        'width': 1280,
        'height': 720,
        'fps': 30,
        'processing_fps': 15,
        'enabled': True
    }
    
    REQUIRED_CAMERA_FIELDS = ['id', 'source', 'width', 'height', 'fps']
    
    PERFORMANCE_MODES = {
        'balanced': {
            'processing_fps': 15,
            'max_faces': 10,
            'batch_size': 4
        },
        'speed': {
            'processing_fps': 30,
            'max_faces': 5,
            'batch_size': 8
        },
        'accuracy': {
            'processing_fps': 10,
            'max_faces': 20,
            'batch_size': 2
        }
    }
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration"""
        self.config_path = config_path
        
        # Setup database path
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        db_dir = data_dir / "db"
        db_dir.mkdir(exist_ok=True)
        db_path = db_dir / "config.db"
        
        # Initialize config database
        self.db = ConfigDB(str(db_path))
        
        # Initialize configuration in correct order
        self._init_defaults()  # First initialize default settings
        self._ensure_default_camera()  # Then ensure default camera exists
        self._migrate_yaml_to_db()  # Then migrate any YAML configs
        self._apply_performance_mode()  # Finally apply performance settings
        
        # Verify default camera after all initialization
        if not self.get_camera(self.DEFAULT_CAMERA['id']):
            logger.warning("Default camera not found after initialization, adding it now")
            self._ensure_default_camera()
    
    def _init_defaults(self):
        """Initialize default configuration values"""
        try:
            for section, values in self.DEFAULT_CONFIG.items():
                current = self.db.get_section(section) or {}
                for key, value in values.items():
                    if key not in current:
                        self.db.set(section, key, value)
            logger.debug("Default configuration initialized")
        except Exception as e:
            logger.error(f"Error initializing defaults: {e}")
    
    def _ensure_default_camera(self):
        """Ensure default camera configuration exists"""
        try:
            existing_camera = self.db.get_camera(self.DEFAULT_CAMERA['id'])
            if not existing_camera:
                logger.info("Adding default camera configuration")
                self.db.add_camera(self.DEFAULT_CAMERA)
            else:
                # Update with any new default fields while preserving existing values
                updated_config = self.DEFAULT_CAMERA.copy()
                for key, value in existing_camera.items():
                    if value is not None:  # Only update if value exists
                        updated_config[key] = value
                self.db.update_camera(self.DEFAULT_CAMERA['id'], updated_config)
                logger.debug(f"Updated default camera configuration: {updated_config}")
        except Exception as e:
            logger.error(f"Error ensuring default camera: {e}")
    
    def _apply_performance_mode(self):
        """Apply performance mode settings"""
        try:
            mode = self.get('detection', 'performance_mode')
            if mode in self.PERFORMANCE_MODES:
                settings = self.PERFORMANCE_MODES[mode]
                for key, value in settings.items():
                    self.set('detection', key, value)
                logger.info(f"Applied performance mode: {mode}")
        except Exception as e:
            logger.error(f"Error applying performance mode: {e}")
    
    def _migrate_yaml_to_db(self):
        """Migrate existing YAML config to SQLite if needed"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                
                if yaml_config:
                    # Migrate each section
                    for section, values in yaml_config.items():
                        if isinstance(values, dict):
                            for key, value in values.items():
                                self.db.set(section, key, value)
                        else:
                            self.db.set('general', section, values)
                    
                    # Migrate cameras if present
                    cameras = yaml_config.get('cameras', [])
                    if isinstance(cameras, list):
                        for camera in cameras:
                            if isinstance(camera, dict) and 'id' in camera:
                                self.db.add_camera(camera)
                    
                    logger.info("Migrated YAML configuration to SQLite")
        except Exception as e:
            logger.error(f"Error migrating YAML config: {e}")
    
    def export_yaml(self, output_path: Optional[str] = None) -> bool:
        """Export configuration to YAML format"""
        try:
            config = self.get_all()
            cameras = self.get_cameras()
            if cameras:
                config['cameras'] = cameras
            
            output_path = output_path or self.config_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Configuration exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def validate_camera_config(self, camera: Dict[str, Any]) -> bool:
        """Validate camera configuration"""
        try:
            # Check required fields
            for field in self.REQUIRED_CAMERA_FIELDS:
                if field not in camera:
                    logger.error(f"Missing required camera field: {field}")
                    return False
            
            # Validate numeric fields
            if not isinstance(camera.get('width'), int) or camera.get('width', 0) <= 0:
                logger.error("Invalid camera width")
                return False
            if not isinstance(camera.get('height'), int) or camera.get('height', 0) <= 0:
                logger.error("Invalid camera height")
                return False
            if not isinstance(camera.get('fps'), int) or camera.get('fps', 0) <= 0:
                logger.error("Invalid camera fps")
                return False
            
            # Validate processing_fps if present
            if 'processing_fps' in camera:
                if not isinstance(camera['processing_fps'], int) or camera['processing_fps'] <= 0:
                    logger.error("Invalid processing_fps")
                    return False
                if camera['processing_fps'] > camera.get('fps', 30):
                    logger.warning("processing_fps is greater than fps, setting to fps value")
                    camera['processing_fps'] = camera['fps']
            
            return True
        except Exception as e:
            logger.error(f"Error validating camera config: {e}")
            return False
    
    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value"""
        if key is None:
            return self.db.get_section(section)
        return self.db.get(section, key, default)
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        self.db.set(section, key, value)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self.db.get_all()
    
    def add_camera(self, camera: Dict[str, Any]) -> bool:
        """Add a camera configuration"""
        try:
            # Ensure all required fields are present
            camera_config = self.DEFAULT_CAMERA.copy()
            camera_config.update(camera)
            
            if not self.validate_camera_config(camera_config):
                return False
                
            success = self.db.add_camera(camera_config)
            if success:
                logger.info(f"Added camera configuration: {camera_config['id']}")
            return success
        except Exception as e:
            logger.error(f"Error adding camera: {e}")
            return False
    
    def update_camera(self, camera_id: str, camera: Dict[str, Any]) -> bool:
        """Update a camera configuration"""
        try:
            existing_camera = self.db.get_camera(camera_id)
            if not existing_camera:
                logger.error(f"Camera not found: {camera_id}")
                return False
            
            # Merge with existing configuration
            updated_config = existing_camera.copy()
            updated_config.update(camera)
            
            if not self.validate_camera_config(updated_config):
                return False
                
            success = self.db.update_camera(camera_id, updated_config)
            if success:
                logger.info(f"Updated camera configuration: {camera_id}")
            return success
        except Exception as e:
            logger.error(f"Error updating camera: {e}")
            return False
    
    def delete_camera(self, camera_id: str) -> bool:
        """Delete a camera configuration"""
        return self.db.delete_camera(camera_id)
    
    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get a camera configuration"""
        try:
            camera = self.db.get_camera(camera_id)
            if camera:
                # Ensure all default fields are present
                default_config = self.DEFAULT_CAMERA.copy()
                default_config.update(camera)
                return default_config
            return None
        except Exception as e:
            logger.error(f"Error getting camera: {e}")
            return None
    
    def get_cameras(self) -> List[Dict[str, Any]]:
        """Get all camera configurations"""
        try:
            # First try to get cameras from database
            cameras = self.db.get_cameras()
            
            # If no cameras found, initialize with default camera
            if not cameras:
                logger.info("No cameras found in database, initializing with default camera")
                try:
                    # Add default camera to database
                    self.add_camera(self.DEFAULT_CAMERA)
                    # Retrieve all cameras again
                    cameras = self.db.get_cameras()
                except Exception as e:
                    logger.error(f"Failed to initialize default camera: {e}")
                    return [self.DEFAULT_CAMERA]  # Return default as last resort
            
            # Validate each camera configuration
            valid_cameras = []
            for camera in cameras:
                try:
                    if self.validate_camera_config(camera):
                        valid_cameras.append(camera)
                    else:
                        logger.warning(f"Invalid camera configuration found for camera {camera.get('id', 'unknown')}")
                except Exception as e:
                    logger.error(f"Error validating camera {camera.get('id', 'unknown')}: {e}")
            
            # If no valid cameras found after validation, return default
            if not valid_cameras:
                logger.warning("No valid cameras found after validation, returning default camera")
                return [self.DEFAULT_CAMERA]
                
            return valid_cameras
            
        except Exception as e:
            logger.error(f"Error getting cameras: {e}")
            # Return default camera as fallback in case of error
            return [self.DEFAULT_CAMERA]
    
    def close(self):
        """Close database connection"""
        self.db.close()
"""
Configuration management using SQLite with advanced monitoring
"""

import os
import yaml
from typing import Any, Dict, List, Optional
from src.core.logger import setup_logger
from src.core.config_db import ConfigDB

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
            'performance_mode': 'balanced'  # balanced, speed, or accuracy
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
            'port': 5000,
            'workers': 1,
            'enable_docs': True,
            'cors_origins': ['*'],
            'rate_limit': 100  # requests per minute
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
        'id': 'mac-builtin',
        'name': 'Mac Built-in Camera',
        'source': '0',  # Use 0 for built-in camera
        'width': 1280,
        'height': 720,
        'fps': 30,
        'processing_fps': 15,
        'enabled': True,
        'auto_reconnect': True,
        'reconnect_delay': 5,  # seconds
        'metrics': {
            'track_fps': True,
            'track_latency': True,
            'track_detection_counts': True,
            'track_recognition_scores': True,
            'track_memory_usage': True,
            'track_cpu_usage': True,
            'track_queue_size': True,
            'track_error_rates': True
        },
        'advanced': {
            'exposure': 'auto',
            'white_balance': 'auto',
            'brightness': 50,
            'contrast': 50,
            'saturation': 50,
            'sharpness': 50,
            'focus': 'auto'
        },
        'recording': {
            'enabled': False,
            'format': 'mp4',
            'quality': 'high',
            'segment_duration': 600,  # 10 minutes
            'max_segments': 144  # 24 hours worth
        },
        'alerts': {
            'on_face_detected': False,
            'on_recognition': False,
            'on_error': True,
            'on_performance_issue': True
        }
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
        self.db = ConfigDB()
        self._init_defaults()
        self._migrate_yaml_to_db()
        self._ensure_default_camera()
        self._apply_performance_mode()
    
    def _init_defaults(self):
        """Initialize default configuration values"""
        try:
            for section, values in self.DEFAULT_CONFIG.items():
                current = self.db.get_section(section)
                for key, value in values.items():
                    if key not in current:
                        self.db.set_value(section, key, value)
            logger.debug("Default configuration initialized")
        except Exception as e:
            logger.error(f"Error initializing defaults: {e}")
    
    def _ensure_default_camera(self):
        """Ensure default camera configuration exists"""
        try:
            existing_camera = self.get_camera(self.DEFAULT_CAMERA['id'])
            if not existing_camera:
                logger.info("Adding default camera configuration")
                self.add_camera(self.DEFAULT_CAMERA)
            else:
                # Update with any new default fields while preserving existing values
                updated_config = self.DEFAULT_CAMERA.copy()
                updated_config.update(existing_camera)
                self.update_camera(self.DEFAULT_CAMERA['id'], updated_config)
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
                                self.db.set_value(section, key, value)
                        else:
                            self.db.set_value('general', section, values)
                    
                    # Migrate cameras if present
                    cameras = yaml_config.get('cameras', [])
                    if isinstance(cameras, list):
                        for camera in cameras:
                            if isinstance(camera, dict) and 'id' in camera:
                                self.add_camera(camera)  # Use add_camera for validation
                    
                    logger.info("Migrated YAML configuration to SQLite")
                    
                    # Backup and remove the YAML file
                    backup_path = self.config_path + '.bak'
                    os.rename(self.config_path, backup_path)
                    logger.info(f"Backed up YAML config to {backup_path}")
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
            if not isinstance(camera['width'], int) or camera['width'] <= 0:
                logger.error("Invalid camera width")
                return False
            if not isinstance(camera['height'], int) or camera['height'] <= 0:
                logger.error("Invalid camera height")
                return False
            if not isinstance(camera['fps'], int) or camera['fps'] <= 0:
                logger.error("Invalid camera fps")
                return False
            
            # Validate processing_fps
            if 'processing_fps' in camera:
                if not isinstance(camera['processing_fps'], int) or camera['processing_fps'] <= 0:
                    logger.error("Invalid processing_fps")
                    return False
                if camera['processing_fps'] > camera['fps']:
                    logger.error("processing_fps cannot be greater than fps")
                    return False
            
            # Validate metrics configuration if present
            if 'metrics' in camera:
                if not isinstance(camera['metrics'], dict):
                    logger.error("Invalid metrics configuration")
                    return False
            
            # Validate advanced settings if present
            if 'advanced' in camera:
                if not isinstance(camera['advanced'], dict):
                    logger.error("Invalid advanced settings")
                    return False
                for key in ['brightness', 'contrast', 'saturation', 'sharpness']:
                    if key in camera['advanced']:
                        value = camera['advanced'][key]
                        if not isinstance(value, (int, float)) or value < 0 or value > 100:
                            logger.error(f"Invalid {key} value")
                            return False
            
            # Validate recording settings if present
            if 'recording' in camera:
                if not isinstance(camera['recording'], dict):
                    logger.error("Invalid recording settings")
                    return False
                if 'segment_duration' in camera['recording']:
                    if not isinstance(camera['recording']['segment_duration'], int) or camera['recording']['segment_duration'] <= 0:
                        logger.error("Invalid segment duration")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating camera config: {e}")
            return False
    
    def get(self, section: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value"""
        if key is None:
            return self.db.get_section(section)
        return self.db.get_value(section, key, default)
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        self.db.set_value(section, key, value)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self.db.get_all()
    
    def add_camera(self, camera: Dict[str, Any]) -> bool:
        """Add a camera configuration"""
        if not self.validate_camera_config(camera):
            return False
        return self.db.add_camera(camera)
    
    def update_camera(self, camera_id: str, camera: Dict[str, Any]) -> bool:
        """Update a camera configuration"""
        if not self.validate_camera_config(camera):
            return False
        return self.db.update_camera(camera_id, camera)
    
    def delete_camera(self, camera_id: str) -> bool:
        """Delete a camera configuration"""
        return self.db.delete_camera(camera_id)
    
    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get a camera configuration"""
        return self.db.get_camera(camera_id)
    
    def get_cameras(self) -> List[Dict[str, Any]]:
        """Get all camera configurations"""
        return self.db.get_cameras()
    
    def close(self):
        """Close database connection"""
        self.db.close()
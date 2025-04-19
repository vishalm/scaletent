"""
Configuration database module for storing and retrieving configuration values
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.logger import setup_logger

logger = setup_logger(__name__)

class ConfigDB:
    """Configuration database for storing and retrieving configuration values"""

    def __init__(self, db_path: str):
        """
        Initialize the configuration database.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
        logger.info(f"Connected to SQLite database: {db_path}")

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                section TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY (section, key)
            )
        """)

        # Drop existing cameras table if it exists (for schema update)
        cursor.execute("DROP TABLE IF EXISTS cameras")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id TEXT PRIMARY KEY,
                name TEXT,
                source TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                fps INTEGER,
                processing_fps INTEGER,
                enabled INTEGER DEFAULT 1
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Database schema initialized")

    def _serialize_value(self, value: Any) -> str:
        """
        Serialize a value to string format for storage.
        
        Args:
            value: Value to serialize
            
        Returns:
            str: Serialized value
        """
        if isinstance(value, (list, dict, bool)):
            return json.dumps(value)
        return str(value)

    def _deserialize_value(self, value: str) -> Any:
        """
        Deserialize a value from string format.
        
        Args:
            value (str): Value to deserialize
            
        Returns:
            Any: Deserialized value
        """
        try:
            # Try to parse as JSON first
            return json.loads(value)
        except json.JSONDecodeError:
            # If not JSON, try to convert to appropriate type
            if value.lower() == 'true':
                return True
            elif value.lower() == 'false':
                return False
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value

    def set(self, section: str, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            value: Configuration value
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            serialized_value = self._serialize_value(value)
            cursor.execute(
                "INSERT OR REPLACE INTO config (section, key, value) VALUES (?, ?, ?)",
                (section, key, serialized_value)
            )
            conn.commit()

        except Exception as e:
            logger.error(f"Error setting value {section}.{key}: {e}")
            raise

        finally:
            conn.close()

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT value FROM config WHERE section = ? AND key = ?",
                (section, key)
            )
            row = cursor.fetchone()

            if row is None:
                return default

            return self._deserialize_value(row[0])

        except Exception as e:
            logger.error(f"Error getting value {section}.{key}: {e}")
            return default

        finally:
            conn.close()

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get all values in a section.
        
        Args:
            section (str): Configuration section
            
        Returns:
            Dict[str, Any]: Section values
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT key, value FROM config WHERE section = ?",
                (section,)
            )
            rows = cursor.fetchall()

            return {
                key: self._deserialize_value(value)
                for key, value in rows
            }

        except Exception as e:
            logger.error(f"Error getting section {section}: {e}")
            return {}

        finally:
            conn.close()

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all configuration values.
        
        Returns:
            Dict[str, Dict[str, Any]]: All configuration values by section
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT section, key, value FROM config")
            rows = cursor.fetchall()

            config = {}
            for section, key, value in rows:
                if section not in config:
                    config[section] = {}
                config[section][key] = self._deserialize_value(value)

            return config

        except Exception as e:
            logger.error(f"Error getting all values: {e}")
            return {}

        finally:
            conn.close()

    def delete(self, section: str, key: str):
        """
        Delete a configuration value.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM config WHERE section = ? AND key = ?",
                (section, key)
            )
            conn.commit()

        except Exception as e:
            logger.error(f"Error deleting value {section}.{key}: {e}")
            raise

        finally:
            conn.close()

    def delete_section(self, section: str):
        """
        Delete an entire configuration section.
        
        Args:
            section (str): Configuration section
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM config WHERE section = ?",
                (section,)
            )
            conn.commit()

        except Exception as e:
            logger.error(f"Error deleting section {section}: {e}")
            raise

        finally:
            conn.close()

    # Camera configuration methods
    def add_camera(self, camera_config: Dict[str, Any]):
        """
        Add or update a camera configuration.
        
        Args:
            camera_config (Dict[str, Any]): Camera configuration
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO cameras 
                (id, name, source, width, height, fps, processing_fps, enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                camera_config["id"],
                camera_config.get("name"),
                camera_config["source"],
                camera_config.get("width"),
                camera_config.get("height"),
                camera_config.get("fps"),
                camera_config.get("processing_fps"),
                1 if camera_config.get("enabled", True) else 0
            ))
            conn.commit()
            logger.info(f"Updated camera: {camera_config['id']}")

        except Exception as e:
            logger.error(f"Error adding camera: {e}")
            raise

        finally:
            conn.close()

    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """
        Get camera configuration.
        
        Args:
            camera_id (str): Camera ID
            
        Returns:
            Optional[Dict[str, Any]]: Camera configuration or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, name, source, width, height, fps, processing_fps, enabled
                FROM cameras WHERE id = ?
            """, (camera_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            return {
                "id": row[0],
                "name": row[1],
                "source": row[2],
                "width": row[3],
                "height": row[4],
                "fps": row[5],
                "processing_fps": row[6],
                "enabled": bool(row[7])
            }

        except Exception as e:
            logger.error(f"Error getting camera {camera_id}: {e}")
            return None

        finally:
            conn.close()

    def get_cameras(self) -> List[Dict[str, Any]]:
        """
        Get all camera configurations.
        
        Returns:
            List[Dict[str, Any]]: List of camera configurations
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM cameras")
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            cameras = []
            for row in rows:
                camera = dict(zip(columns, row))
                # Convert enabled to boolean
                camera['enabled'] = bool(camera['enabled'])
                cameras.append(camera)

            return cameras

        except Exception as e:
            logger.error(f"Error getting cameras: {e}")
            return []

        finally:
            conn.close()

    def delete_camera(self, camera_id: str):
        """
        Delete a camera configuration.
        
        Args:
            camera_id (str): Camera ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
            conn.commit()

        except Exception as e:
            logger.error(f"Error deleting camera {camera_id}: {e}")
            raise

        finally:
            conn.close() 
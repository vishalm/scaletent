"""
SQLite database manager for metrics and data storage
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from src.core.logger import setup_logger

logger = setup_logger(__name__)

class DatabaseManager:
    """Database manager for metrics and data storage"""
    
    def __init__(self, db_path: str = "data/app.db"):
        """Initialize database manager"""
        self.db_path = db_path
        self._ensure_db_dir()
        self.conn = None
        self.cursor = None
        self._connect()
        self._init_schema()
    
    def _ensure_db_dir(self):
        """Ensure database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
    
    def _connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to SQLite database: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def _init_schema(self):
        """Initialize database schema"""
        try:
            schema_path = Path(__file__).parent / "schema.sql"
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            self.cursor.executescript(schema_sql)
            self.conn.commit()
            logger.info("Database schema initialized")
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            raise
    
    def record_performance_metrics(self, camera_id: str, metrics: Dict[str, Any]):
        """Record performance metrics"""
        try:
            self.cursor.execute("""
                INSERT INTO performance_metrics 
                (camera_id, cpu_usage, memory_usage, fps, processing_time, 
                queue_size, faces_detected, faces_recognized)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                camera_id,
                metrics.get('cpu_usage'),
                metrics.get('memory_usage'),
                metrics.get('fps'),
                metrics.get('processing_time'),
                metrics.get('queue_size'),
                metrics.get('faces_detected'),
                metrics.get('faces_recognized')
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error recording performance metrics: {e}")
    
    def record_recognition_metrics(self, camera_id: str, metrics: Dict[str, Any]):
        """Record recognition metrics"""
        try:
            self.cursor.execute("""
                INSERT INTO recognition_metrics 
                (camera_id, person_id, confidence_score, detection_time,
                recognition_time, embedding_generation_time)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                camera_id,
                metrics.get('person_id'),
                metrics.get('confidence_score'),
                metrics.get('detection_time'),
                metrics.get('recognition_time'),
                metrics.get('embedding_generation_time')
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error recording recognition metrics: {e}")
    
    def start_recording(self, camera_id: str, file_path: str, format: str = 'mp4', quality: str = 'high') -> int:
        """Start a new recording"""
        try:
            self.cursor.execute("""
                INSERT INTO recordings 
                (camera_id, file_path, format, quality, status)
                VALUES (?, ?, ?, ?, 'recording')
            """, (camera_id, file_path, format, quality))
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return None
    
    def add_recording_segment(self, recording_id: int, segment_number: int, file_path: str) -> bool:
        """Add a recording segment"""
        try:
            self.cursor.execute("""
                INSERT INTO recording_segments 
                (recording_id, segment_number, file_path, status)
                VALUES (?, ?, ?, 'recording')
            """, (recording_id, segment_number, file_path))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding recording segment: {e}")
            return False
    
    def complete_recording(self, recording_id: int, duration: int, file_size: int):
        """Complete a recording"""
        try:
            self.cursor.execute("""
                UPDATE recordings 
                SET status = 'completed', 
                    end_time = CURRENT_TIMESTAMP,
                    duration = ?,
                    file_size = ?
                WHERE id = ?
            """, (duration, file_size, recording_id))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error completing recording: {e}")
    
    def record_face_detection(self, camera_id: str, detection: Dict[str, Any], embedding: Optional[np.ndarray] = None):
        """Record a face detection"""
        try:
            if embedding is not None:
                embedding_bytes = embedding.tobytes()
            else:
                embedding_bytes = None
            
            self.cursor.execute("""
                INSERT INTO face_detections 
                (camera_id, frame_id, bbox_x, bbox_y, bbox_width, bbox_height,
                confidence, person_id, embedding, thumbnail_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                camera_id,
                detection.get('frame_id'),
                detection.get('bbox_x'),
                detection.get('bbox_y'),
                detection.get('bbox_width'),
                detection.get('bbox_height'),
                detection.get('confidence'),
                detection.get('person_id'),
                embedding_bytes,
                detection.get('thumbnail_path')
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error recording face detection: {e}")
    
    def record_event(self, camera_id: str, event_type: str, details: Dict[str, Any], severity: str = 'info'):
        """Record an event"""
        try:
            self.cursor.execute("""
                INSERT INTO events 
                (camera_id, event_type, details, severity)
                VALUES (?, ?, ?, ?)
            """, (camera_id, event_type, json.dumps(details), severity))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error recording event: {e}")
    
    def record_system_health(self, metrics: Dict[str, Any]):
        """Record system health metrics"""
        try:
            self.cursor.execute("""
                INSERT INTO system_health 
                (cpu_usage, memory_usage, disk_usage, temperature,
                uptime, active_cameras, total_fps, error_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.get('cpu_usage'),
                metrics.get('memory_usage'),
                metrics.get('disk_usage'),
                metrics.get('temperature'),
                metrics.get('uptime'),
                metrics.get('active_cameras'),
                metrics.get('total_fps'),
                metrics.get('error_count')
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error recording system health: {e}")
    
    def log_error(self, error_type: str, error_message: str, stack_trace: str = None,
                camera_id: str = None, severity: str = 'error'):
        """Log an error"""
        try:
            self.cursor.execute("""
                INSERT INTO error_logs 
                (error_type, error_message, stack_trace, camera_id, severity)
                VALUES (?, ?, ?, ?, ?)
            """, (error_type, error_message, stack_trace, camera_id, severity))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error logging error: {e}")
    
    def record_maintenance(self, operation_type: str, details: Dict[str, Any], 
                        status: str, duration: int, bytes_affected: int):
        """Record maintenance operation"""
        try:
            self.cursor.execute("""
                INSERT INTO maintenance_logs 
                (operation_type, details, status, duration, bytes_affected)
                VALUES (?, ?, ?, ?, ?)
            """, (operation_type, json.dumps(details), status, duration, bytes_affected))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error recording maintenance: {e}")
    
    def get_camera_performance(self, camera_id: str, time_range: int = 3600) -> Dict[str, Any]:
        """Get camera performance metrics for the last time_range seconds"""
        try:
            self.cursor.execute("""
                SELECT * FROM v_camera_performance
                WHERE camera_id = ?
            """, (camera_id,))
            return dict(self.cursor.fetchone())
        except Exception as e:
            logger.error(f"Error getting camera performance: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 7):
        """Clean up old data"""
        try:
            tables = ['performance_metrics', 'recognition_metrics', 'events', 
                     'face_detections', 'system_health', 'error_logs']
            
            for table in tables:
                self.cursor.execute(f"""
                    DELETE FROM {table}
                    WHERE timestamp < datetime('now', '-{days_to_keep} days')
                """)
            
            self.conn.commit()
            logger.info(f"Cleaned up data older than {days_to_keep} days")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def reset_camera_stats(self, camera_id: str) -> bool:
        """Reset all statistics for a specific camera"""
        try:
            tables = [
                'performance_metrics',
                'recognition_metrics',
                'face_detections',
                'events',
                'error_logs'
            ]
            
            for table in tables:
                self.cursor.execute(f"""
                    DELETE FROM {table}
                    WHERE camera_id = ?
                """, (camera_id,))
            
            # Record the reset event
            self.record_event(
                camera_id=camera_id,
                event_type='stats_reset',
                details={'timestamp': datetime.now().isoformat()},
                severity='info'
            )
            
            self.conn.commit()
            logger.info(f"Reset statistics for camera {camera_id}")
            return True
        except Exception as e:
            logger.error(f"Error resetting camera stats: {e}")
            return False
    
    def reset_all_stats(self) -> bool:
        """Reset all statistics for all cameras"""
        try:
            tables = [
                'performance_metrics',
                'recognition_metrics',
                'face_detections',
                'events',
                'system_health',
                'error_logs'
            ]
            
            for table in tables:
                self.cursor.execute(f"DELETE FROM {table}")
            
            # Record the reset event in maintenance logs
            self.record_maintenance(
                operation_type='stats_reset',
                details={'timestamp': datetime.now().isoformat()},
                status='completed',
                duration=0,
                bytes_affected=0
            )
            
            self.conn.commit()
            logger.info("Reset all statistics")
            return True
        except Exception as e:
            logger.error(f"Error resetting all stats: {e}")
            return False
    
    def reset_error_logs(self, camera_id: Optional[str] = None) -> bool:
        """Reset error logs, optionally for a specific camera"""
        try:
            if camera_id:
                self.cursor.execute("""
                    DELETE FROM error_logs
                    WHERE camera_id = ?
                """, (camera_id,))
                logger.info(f"Reset error logs for camera {camera_id}")
            else:
                self.cursor.execute("DELETE FROM error_logs")
                logger.info("Reset all error logs")
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error resetting error logs: {e}")
            return False
    
    def reset_performance_metrics(self, camera_id: Optional[str] = None, older_than_days: Optional[int] = None) -> bool:
        """Reset performance metrics with optional filters"""
        try:
            if camera_id and older_than_days:
                self.cursor.execute("""
                    DELETE FROM performance_metrics
                    WHERE camera_id = ?
                    AND timestamp < datetime('now', '-? days')
                """, (camera_id, older_than_days))
            elif camera_id:
                self.cursor.execute("""
                    DELETE FROM performance_metrics
                    WHERE camera_id = ?
                """, (camera_id,))
            elif older_than_days:
                self.cursor.execute("""
                    DELETE FROM performance_metrics
                    WHERE timestamp < datetime('now', '-? days')
                """, (older_than_days,))
            else:
                self.cursor.execute("DELETE FROM performance_metrics")
            
            self.conn.commit()
            logger.info(f"Reset performance metrics (camera_id={camera_id}, older_than_days={older_than_days})")
            return True
        except Exception as e:
            logger.error(f"Error resetting performance metrics: {e}")
            return False
    
    def get_stats_summary(self, camera_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of current statistics"""
        try:
            result = {}
            
            # Get counts from various tables
            tables = [
                'performance_metrics',
                'recognition_metrics',
                'face_detections',
                'events',
                'error_logs'
            ]
            
            for table in tables:
                if camera_id:
                    self.cursor.execute(f"""
                        SELECT COUNT(*) as count,
                               MIN(timestamp) as oldest,
                               MAX(timestamp) as newest
                        FROM {table}
                        WHERE camera_id = ?
                    """, (camera_id,))
                else:
                    self.cursor.execute(f"""
                        SELECT COUNT(*) as count,
                               MIN(timestamp) as oldest,
                               MAX(timestamp) as newest
                        FROM {table}
                    """)
                
                row = self.cursor.fetchone()
                result[table] = {
                    'count': row['count'],
                    'oldest': row['oldest'],
                    'newest': row['newest']
                }
            
            # Get storage usage
            if camera_id:
                self.cursor.execute("""
                    SELECT SUM(file_size) as total_size
                    FROM recordings
                    WHERE camera_id = ?
                """, (camera_id,))
            else:
                self.cursor.execute("""
                    SELECT SUM(file_size) as total_size
                    FROM recordings
                """)
            
            row = self.cursor.fetchone()
            result['storage'] = {
                'total_size': row['total_size'] or 0,
                'unit': 'bytes'
            }
            
            return result
        except Exception as e:
            logger.error(f"Error getting stats summary: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed") 
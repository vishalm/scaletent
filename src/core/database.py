"""
Database management for ScaleTent
"""

import os
import json
import pickle
import time
import motor.motor_asyncio
from pathlib import Path
from datetime import datetime, timedelta
import asyncio

from core.logger import setup_logger

logger = setup_logger(__name__)

class Database:
    """
    Database management class
    
    Supports multiple backends:
    - MongoDB
    - Redis
    - File-based
    """
    
    def __init__(self, config):
        """
        Initialize database connection
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.db_type = config.get("storage.database.type", "file")
        self.client = None
        self.db = None
        
        logger.info(f"Initializing database with type: {self.db_type}")
        
    async def connect(self):
        """
        Connect to database
        
        Returns:
            bool: Success status
        """
        try:
            if self.db_type == "mongodb":
                return await self._connect_mongodb()
            elif self.db_type == "file":
                return self._connect_file()
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
        
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False
    
    async def _connect_mongodb(self):
        """
        Connect to MongoDB
        
        Returns:
            bool: Success status
        """
        try:
            host = self.config.get("storage.database.host", "localhost")
            port = self.config.get("storage.database.port", 27017)
            username = self.config.get("storage.database.username")
            password = self.config.get("storage.database.password")
            db_name = self.config.get("storage.database.name", "scaletent")
            
            # Build connection string
            if username and password:
                conn_str = f"mongodb://{username}:{password}@{host}:{port}/{db_name}"
            else:
                conn_str = f"mongodb://{host}:{port}/{db_name}"
            
            logger.info(f"Connecting to MongoDB at {host}:{port}")
            
            # Connect to MongoDB
            self.client = motor.motor_asyncio.AsyncIOMotorClient(conn_str)
            self.db = self.client[db_name]
            
            # Test connection
            await self.db.command("ping")
            
            logger.info("MongoDB connection successful")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            return False
    
    def _connect_file(self):
        """
        Set up file-based database
        
        Returns:
            bool: Success status
        """
        try:
            data_path = self.config.get("storage.data_path", "data")
            
            # Ensure data directory exists
            os.makedirs(data_path, exist_ok=True)
            
            # Set up collections as directories
            self.detections_path = os.path.join(data_path, "detections")
            self.profiles_path = os.path.join(data_path, "profiles")
            self.analytics_path = os.path.join(data_path, "analytics")
            
            os.makedirs(self.detections_path, exist_ok=True)
            os.makedirs(self.profiles_path, exist_ok=True)
            os.makedirs(self.analytics_path, exist_ok=True)
            
            logger.info("File-based database setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up file-based database: {e}")
            return False
    
    async def disconnect(self):
        """Close database connection"""
        if self.db_type == "mongodb" and self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def store_detection(self, detection_data):
        """
        Store detection data
        
        Args:
            detection_data (dict): Detection data to store
        
        Returns:
            str: ID of stored detection or None if error
        """
        try:
            # Generate ID if not present
            if "_id" not in detection_data:
                detection_data["_id"] = f"detection-{int(time.time() * 1000)}"
            
            # Add timestamp if not present
            if "timestamp" not in detection_data:
                detection_data["timestamp"] = datetime.utcnow().isoformat()
            
            if self.db_type == "mongodb":
                result = await self.db.detections.insert_one(detection_data)
                logger.debug(f"Detection stored with ID: {result.inserted_id}")
                return str(result.inserted_id)
            
            elif self.db_type == "file":
                detection_id = detection_data["_id"]
                file_path = os.path.join(self.detections_path, f"{detection_id}.json")
                
                with open(file_path, 'w') as f:
                    json.dump(detection_data, f)
                
                logger.debug(f"Detection stored at: {file_path}")
                return detection_id
        
        except Exception as e:
            logger.error(f"Error storing detection: {e}")
            return None
    
    async def get_detection(self, detection_id):
        """
        Get detection data by ID
        
        Args:
            detection_id (str): Detection ID
        
        Returns:
            dict: Detection data or None if not found
        """
        try:
            if self.db_type == "mongodb":
                detection = await self.db.detections.find_one({"_id": detection_id})
                return detection
            
            elif self.db_type == "file":
                file_path = os.path.join(self.detections_path, f"{detection_id}.json")
                
                if not os.path.exists(file_path):
                    logger.warning(f"Detection not found: {detection_id}")
                    return None
                
                with open(file_path, 'r') as f:
                    detection = json.load(f)
                
                return detection
        
        except Exception as e:
            logger.error(f"Error getting detection: {e}")
            return None
    
    async def get_recent_detections(self, limit=100, camera_id=None):
        """
        Get recent detections
        
        Args:
            limit (int): Maximum number of detections to return
            camera_id (str, optional): Filter by camera ID
        
        Returns:
            list: List of detection data
        """
        try:
            if self.db_type == "mongodb":
                query = {}
                if camera_id:
                    query["camera_id"] = camera_id
                
                cursor = self.db.detections.find(query)
                cursor = cursor.sort("timestamp", -1).limit(limit)
                
                return await cursor.to_list(length=limit)
            
            elif self.db_type == "file":
                detections = []
                
                for filename in os.listdir(self.detections_path):
                    if not filename.endswith('.json'):
                        continue
                    
                    file_path = os.path.join(self.detections_path, filename)
                    
                    with open(file_path, 'r') as f:
                        detection = json.load(f)
                    
                    if camera_id and detection.get("camera_id") != camera_id:
                        continue
                    
                    detections.append(detection)
                
                # Sort by timestamp (descending)
                detections.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                
                # Limit results
                return detections[:limit]
        
        except Exception as e:
            logger.error(f"Error getting recent detections: {e}")
            return []
    
    async def store_profile(self, profile_data):
        """
        Store profile data
        
        Args:
            profile_data (dict): Profile data to store
        
        Returns:
            str: ID of stored profile or None if error
        """
        try:
            # Ensure ID is present
            if "id" not in profile_data:
                logger.error("Profile data must have an ID")
                return None
            
            profile_id = profile_data["id"]
            
            # Add registration time if not present
            if "registration_time" not in profile_data:
                profile_data["registration_time"] = datetime.utcnow().isoformat()
            
            if self.db_type == "mongodb":
                # Use upsert to update if exists or insert if not
                result = await self.db.profiles.update_one(
                    {"id": profile_id},
                    {"$set": profile_data},
                    upsert=True
                )
                
                logger.debug(f"Profile stored with ID: {profile_id}")
                return profile_id
            
            elif self.db_type == "file":
                file_path = os.path.join(self.profiles_path, f"{profile_id}.json")
                
                with open(file_path, 'w') as f:
                    json.dump(profile_data, f)
                
                logger.debug(f"Profile stored at: {file_path}")
                return profile_id
        
        except Exception as e:
            logger.error(f"Error storing profile: {e}")
            return None
    
    async def get_profile(self, profile_id):
        """
        Get profile data by ID
        
        Args:
            profile_id (str): Profile ID
        
        Returns:
            dict: Profile data or None if not found
        """
        try:
            if self.db_type == "mongodb":
                profile = await self.db.profiles.find_one({"id": profile_id})
                return profile
            
            elif self.db_type == "file":
                file_path = os.path.join(self.profiles_path, f"{profile_id}.json")
                
                if not os.path.exists(file_path):
                    logger.warning(f"Profile not found: {profile_id}")
                    return None
                
                with open(file_path, 'r') as f:
                    profile = json.load(f)
                
                return profile
        
        except Exception as e:
            logger.error(f"Error getting profile: {e}")
            return None
    
    async def delete_profile(self, profile_id):
        """
        Delete profile by ID
        
        Args:
            profile_id (str): Profile ID
        
        Returns:
            bool: Success status
        """
        try:
            if self.db_type == "mongodb":
                result = await self.db.profiles.delete_one({"id": profile_id})
                success = result.deleted_count > 0
                
                if success:
                    logger.info(f"Profile deleted: {profile_id}")
                else:
                    logger.warning(f"Profile not found for deletion: {profile_id}")
                
                return success
            
            elif self.db_type == "file":
                file_path = os.path.join(self.profiles_path, f"{profile_id}.json")
                
                if not os.path.exists(file_path):
                    logger.warning(f"Profile not found for deletion: {profile_id}")
                    return False
                
                os.remove(file_path)
                logger.info(f"Profile deleted: {profile_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error deleting profile: {e}")
            return False
    
    async def get_all_profiles(self):
        """
        Get all profiles
        
        Returns:
            list: List of profile data
        """
        try:
            if self.db_type == "mongodb":
                cursor = self.db.profiles.find()
                return await cursor.to_list(length=None)
            
            elif self.db_type == "file":
                profiles = []
                
                for filename in os.listdir(self.profiles_path):
                    if not filename.endswith('.json'):
                        continue
                    
                    file_path = os.path.join(self.profiles_path, filename)
                    
                    with open(file_path, 'r') as f:
                        profile = json.load(f)
                    
                    profiles.append(profile)
                
                return profiles
        
        except Exception as e:
            logger.error(f"Error getting all profiles: {e}")
            return []
    
    async def store_analytics(self, analytics_data):
        """
        Store analytics data
        
        Args:
            analytics_data (dict): Analytics data to store
        
        Returns:
            str: ID of stored analytics or None if error
        """
        try:
            # Generate ID if not present
            if "_id" not in analytics_data:
                analytics_data["_id"] = f"analytics-{int(time.time() * 1000)}"
            
            # Add timestamp if not present
            if "timestamp" not in analytics_data:
                analytics_data["timestamp"] = datetime.utcnow().isoformat()
            
            if self.db_type == "mongodb":
                result = await self.db.analytics.insert_one(analytics_data)
                logger.debug(f"Analytics stored with ID: {result.inserted_id}")
                return str(result.inserted_id)
            
            elif self.db_type == "file":
                analytics_id = analytics_data["_id"]
                file_path = os.path.join(self.analytics_path, f"{analytics_id}.json")
                
                with open(file_path, 'w') as f:
                    json.dump(analytics_data, f)
                
                logger.debug(f"Analytics stored at: {file_path}")
                return analytics_id
        
        except Exception as e:
            logger.error(f"Error storing analytics: {e}")
            return None
    
    async def get_analytics(self, start_time=None, end_time=None):
        """
        Get analytics data between time range
        
        Args:
            start_time (str, optional): Start time in ISO format
            end_time (str, optional): End time in ISO format
        
        Returns:
            list: List of analytics data
        """
        try:
            if start_time is None:
                # Default to 24 hours ago
                start_time = (datetime.utcnow() - timedelta(days=1)).isoformat()
            
            if end_time is None:
                end_time = datetime.utcnow().isoformat()
            
            if self.db_type == "mongodb":
                query = {
                    "timestamp": {
                        "$gte": start_time,
                        "$lte": end_time
                    }
                }
                
                cursor = self.db.analytics.find(query)
                cursor = cursor.sort("timestamp", 1)
                
                return await cursor.to_list(length=None)
            
            elif self.db_type == "file":
                analytics = []
                
                for filename in os.listdir(self.analytics_path):
                    if not filename.endswith('.json'):
                        continue
                    
                    file_path = os.path.join(self.analytics_path, filename)
                    
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    timestamp = data.get("timestamp", "")
                    
                    if timestamp >= start_time and timestamp <= end_time:
                        analytics.append(data)
                
                # Sort by timestamp (ascending)
                analytics.sort(key=lambda x: x.get("timestamp", ""))
                
                return analytics
        
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return []
    
    async def cleanup_old_data(self):
        """
        Clean up old data based on retention policy
        
        Returns:
            int: Number of records deleted
        """
        try:
            # Get retention days from config
            detection_retention_days = self.config.get("privacy.data_retention.detection_data_days", 30)
            
            # Calculate cutoff date
            cutoff_date = (datetime.utcnow() - timedelta(days=detection_retention_days)).isoformat()
            
            logger.info(f"Cleaning up data older than {cutoff_date}")
            
            deleted_count = 0
            
            if self.db_type == "mongodb":
                # Delete old detections
                result = await self.db.detections.delete_many({
                    "timestamp": {"$lt": cutoff_date}
                })
                
                deleted_count = result.deleted_count
                
            elif self.db_type == "file":
                # Delete old detection files
                for filename in os.listdir(self.detections_path):
                    if not filename.endswith('.json'):
                        continue
                    
                    file_path = os.path.join(self.detections_path, filename)
                    
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        timestamp = data.get("timestamp", "")
                        
                        if timestamp < cutoff_date:
                            os.remove(file_path)
                            deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
            
            logger.info(f"Deleted {deleted_count} old detection records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
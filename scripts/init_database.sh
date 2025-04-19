#!/usr/bin/env python3
"""
ScaleTent - Database Initialization Script
Initializes the database and creates necessary collections
"""

import os
import sys
import argparse
import json
import asyncio
import yaml
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from core.config import Config
from core.database import Database
from core.logger import setup_logger

# Set up logger
logger = setup_logger("db_init")

# Parse arguments
parser = argparse.ArgumentParser(description='Initialize database for ScaleTent')
parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
parser.add_argument('--sample-data', action='store_true', help='Load sample data')
parser.add_argument('--force', action='store_true', help='Force reinitialization even if database exists')
args = parser.parse_args()

# Sample profiles for initialization
SAMPLE_PROFILES = [
    {
        "id": "JS001",
        "name": "John Smith",
        "role": "Attendee",
        "organization": "Example Corp",
        "registration_time": datetime.utcnow().isoformat(),
        "additional_info": {
            "email": "john.smith@example.com",
            "phone": "+1-555-123-4567",
            "badge_id": "A12345"
        }
    },
    {
        "id": "AJ002",
        "name": "Alice Johnson",
        "role": "Speaker",
        "organization": "Tech Innovations",
        "registration_time": datetime.utcnow().isoformat(),
        "additional_info": {
            "email": "alice.johnson@techinnovations.example",
            "phone": "+1-555-987-6543",
            "badge_id": "S54321",
            "presentation": "Next-Gen AI Systems"
        }
    },
    {
        "id": "RB003",
        "name": "Robert Brown",
        "role": "Staff",
        "organization": "Event Management",
        "registration_time": datetime.utcnow().isoformat(),
        "additional_info": {
            "email": "robert.brown@eventmanagement.example",
            "phone": "+1-555-456-7890",
            "badge_id": "STF001",
            "position": "Event Coordinator"
        }
    }
]

# Sample camera configurations
SAMPLE_CAMERAS = [
    {
        "id": "main-entrance-01",
        "name": "Main Entrance Camera",
        "source": "0",  # Use default camera
        "width": 1280,
        "height": 720,
        "fps": 30,
        "enabled": True
    },
    {
        "id": "reception-desk-01",
        "name": "Reception Desk Camera",
        "source": "1",  # Use second camera if available
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "enabled": False  # Disabled by default
    }
]

async def init_mongodb(db, force=False):
    """Initialize MongoDB collections"""
    logger.info("Initializing MongoDB collections")
    
    # Check if database already has collections
    existing_collections = await db.db.list_collection_names()
    
    if existing_collections and not force:
        logger.info(f"MongoDB already has collections: {existing_collections}")
        logger.info("Use --force to reinitialize")
        return False
    
    # Drop existing collections if force is enabled
    if force and existing_collections:
        logger.warning("Dropping existing collections")
        for collection in existing_collections:
            await db.db.drop_collection(collection)
    
    # Create indices for detections collection
    await db.db.detections.create_index("timestamp")
    await db.db.detections.create_index("camera_id")
    
    # Create indices for profiles collection
    await db.db.profiles.create_index("id", unique=True)
    
    # Create indices for analytics collection
    await db.db.analytics.create_index("timestamp")
    
    logger.info("MongoDB collections initialized")
    return True

async def init_file_db(db, force=False):
    """Initialize file-based database directories"""
    logger.info("Initializing file-based database directories")
    
    # Create directories
    os.makedirs(db.detections_path, exist_ok=True)
    os.makedirs(db.profiles_path, exist_ok=True)
    os.makedirs(db.analytics_path, exist_ok=True)
    
    # Clear existing files if force is enabled
    if force:
        logger.warning("Clearing existing files")
        for file in os.listdir(db.detections_path):
            if file.endswith(".json"):
                os.remove(os.path.join(db.detections_path, file))
        
        for file in os.listdir(db.profiles_path):
            if file.endswith(".json"):
                os.remove(os.path.join(db.profiles_path, file))
        
        for file in os.listdir(db.analytics_path):
            if file.endswith(".json"):
                os.remove(os.path.join(db.analytics_path, file))
    
    logger.info("File-based database directories initialized")
    return True

async def load_sample_data(db):
    """Load sample data into database"""
    logger.info("Loading sample data")
    
    # Load sample profiles
    for profile in SAMPLE_PROFILES:
        await db.store_profile(profile)
        logger.info(f"Added profile: {profile['name']} ({profile['id']})")
    
    # Load sample analytics
    now = datetime.utcnow()
    for i in range(24):
        # Create hourly analytics for the past 24 hours
        timestamp = (now.replace(minute=0, second=0, microsecond=0) - 
                     asyncio.timedelta(hours=i))
        
        analytics_data = {
            "timestamp": timestamp.isoformat(),
            "people_count": 10 + i % 5,  # Random variation
            "recognized_count": 5 + i % 3,
            "cameras": {
                "main-entrance-01": {
                    "people_count": 7 + i % 4,
                    "recognized_count": 3 + i % 2
                },
                "reception-desk-01": {
                    "people_count": 3 + i % 2,
                    "recognized_count": 2 + i % 2
                }
            }
        }
        
        await db.store_analytics(analytics_data)
    
    logger.info("Sample data loaded")

async def update_config_cameras(config_path, cameras):
    """Update config file with camera settings"""
    try:
        # Load existing config
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Update cameras section
        config_data['cameras'] = cameras
        
        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        logger.info(f"Updated camera configuration in {config_path}")
        
    except Exception as e:
        logger.error(f"Error updating config file: {e}")

async def main():
    """Main function"""
    try:
        logger.info("===== ScaleTent Database Initialization =====")
        
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        
        if not os.path.exists(args.config):
            logger.error(f"Config file not found: {args.config}")
            return 1
        
        config = Config(config_path=args.config)
        
        # Initialize database
        db = Database(config)
        connected = await db.connect()
        
        if not connected:
            logger.error("Failed to connect to database")
            return 1
        
        # Initialize database structure
        if db.db_type == "mongodb":
            success = await init_mongodb(db, args.force)
        else:
            success = await init_file_db(db, args.force)
        
        if not success and not args.force:
            logger.info("Database already initialized. Use --force to reinitialize.")
            await db.disconnect()
            return 0
        
        # Load sample data if requested
        if args.sample_data:
            await load_sample_data(db)
            
            # Update config with sample cameras
            await update_config_cameras(args.config, SAMPLE_CAMERAS)
        
        await db.disconnect()
        
        logger.info("===== Database Initialization Complete =====")
        return 0
        
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
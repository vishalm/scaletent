"""
API routes for ScaleTent
"""

import asyncio
import json
import base64
import cv2
import numpy as np
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from fastapi import APIRouter, HTTPException, Depends, Query, Path as PathParam, Body
from fastapi import File, UploadFile, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from src.core.logger import setup_logger
from src.core.config import Config
from src.core.camera import Camera
from src.core.db_manager import DatabaseManager

logger = setup_logger(__name__)

# Create router
router = APIRouter(
    prefix="/api",
    tags=["api"],
    responses={404: {"description": "Not found"}},
)

# ======== Models ========

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class PersonData(BaseModel):
    name: Optional[str] = None
    id: str
    role: Optional[str] = None
    registration_time: Optional[str] = None
    organization: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class DetectionResult(BaseModel):
    id: str
    type: str
    confidence: float
    bbox: List[int]
    recognized: bool
    person_data: Optional[PersonData] = None


class DetectionResponse(BaseModel):
    timestamp: str
    frame_id: int
    camera_id: str
    detections: List[DetectionResult]
    analytics: Dict[str, Any]


class RegistrationRequest(BaseModel):
    person_data: PersonData
    image_base64: Optional[str] = None


class CameraInfo(BaseModel):
    id: str
    name: Optional[str] = None
    source: Union[str, int]
    width: int
    height: int
    fps: int
    running: bool = False
    frame_count: int = 0


class SystemStatus(BaseModel):
    status: str
    version: str
    uptime: float
    cameras: Dict[str, Dict[str, Any]]
    detection_fps: float
    connected_clients: int


class ExportRequest(BaseModel):
    data_type: str = "detections"
    format: str = "json"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    camera_id: Optional[str] = None


# ======== Dependencies ========

def get_app_context():
    """Get application context"""
    from main import get_app_context
    return get_app_context()


def get_detector(request: Request):
    """Dependency to get detector instance"""
    context = get_app_context()
    return context.get('detector')


def get_face_detector(request: Request):
    """Dependency to get face detector instance"""
    context = get_app_context()
    return context.get('face_detector')


def get_face_matcher(request: Request):
    """Dependency to get face matcher instance"""
    context = get_app_context()
    return context.get('face_matcher')


def get_camera_manager(request: Request):
    """Dependency to get camera manager instance"""
    if not hasattr(request.app.state, 'components') or 'camera_manager' not in request.app.state.components:
        raise HTTPException(
            status_code=503,
            detail="Camera manager not initialized"
        )
    return request.app.state.components['camera_manager']


def get_publisher(request: Request):
    """Dependency to get publisher instance"""
    if not hasattr(request.app.state, 'components') or 'publisher' not in request.app.state.components:
        raise HTTPException(
            status_code=503,
            detail="Publisher not initialized"
        )
    return request.app.state.components['publisher']


def get_database(request: Request):
    """Dependency to get database instance"""
    if not hasattr(request.app.state, 'components') or 'database' not in request.app.state.components:
        raise HTTPException(
            status_code=503,
            detail="Database not initialized"
        )
    return request.app.state.components['database']


def get_config(request: Request) -> Config:
    """Dependency to get config instance"""
    if not hasattr(request.app.state, 'config'):
        raise HTTPException(
            status_code=503,
            detail="Configuration not initialized"
        )
    return request.app.state.config


def get_api_key(request: Request) -> bool:
    """Dependency to check API key"""
    # TODO: Implement API key validation
    return True


# API Key security (if enabled)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(
    api_key_header: str = Depends(api_key_header),
    config: Config = Depends(get_config)
):
    """Validate API key if security is enabled"""
    if not config.get("api.security.enabled", False):
        return True
    
    valid_api_key = config.get("api.security.api_key")
    if not valid_api_key:
        return True
    
    if api_key_header != valid_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return True


# ======== Routes ========

@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get system status."""
    return {
        "status": "running",
        "version": "0.1.0"
    }


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get system configuration."""
    return {
        "detection": {
            "enabled": True,
            "confidence_threshold": 0.5
        },
        "recognition": {
            "enabled": True,
            "similarity_threshold": 0.7
        }
    }


@router.get("/detections/latest", response_model=DetectionResponse)
async def get_latest_detections(
    camera_id: Optional[str] = Query(None, description="Camera ID to get detections from"),
    api_key: bool = Depends(get_api_key),
    publisher = Depends(get_publisher)
):
    """Get latest detection results"""
    try:
        # Get the most recent message from publisher
        if not publisher.recent_messages:
            return JSONResponse(
                status_code=404,
                content={"detail": "No detection data available"}
            )
        
        # Get the latest message
        latest_message = publisher.recent_messages[-1]
        data = json.loads(latest_message)
        
        # Filter by camera_id if provided
        if camera_id and data.get("camera_id") != camera_id:
            # Search for the latest message from this camera
            for message in reversed(publisher.recent_messages):
                message_data = json.loads(message)
                if message_data.get("camera_id") == camera_id:
                    data = message_data
                    break
            
            # If no message found for this camera
            if data.get("camera_id") != camera_id:
                return JSONResponse(
                    status_code=404,
                    content={"detail": f"No detection data available for camera {camera_id}"}
                )
        
        return data
        
    except Exception as e:
        logger.error(f"Error getting latest detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detections", response_model=List[DetectionResponse])
async def get_detections(
    limit: int = Query(10, ge=1, le=100, description="Number of detection results to return"),
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    start_time: Optional[str] = Query(None, description="Start time in ISO format"),
    end_time: Optional[str] = Query(None, description="End time in ISO format"),
    api_key: bool = Depends(get_api_key),
    database = Depends(get_database)
):
    """Get historical detection results"""
    try:
        # Convert start/end times to timestamps if provided
        start_timestamp = None
        end_timestamp = None
        
        if start_time:
            try:
                start_timestamp = datetime.fromisoformat(start_time).isoformat()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_time format. Use ISO format (YYYY-MM-DDTHH:MM:SS.sssZ)")
        
        if end_time:
            try:
                end_timestamp = datetime.fromisoformat(end_time).isoformat()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_time format. Use ISO format (YYYY-MM-DDTHH:MM:SS.sssZ)")
        
        # Get detections from database
        query = {}
        
        if camera_id:
            query["camera_id"] = camera_id
        
        if start_timestamp or end_timestamp:
            query["timestamp"] = {}
            
            if start_timestamp:
                query["timestamp"]["$gte"] = start_timestamp
            
            if end_timestamp:
                query["timestamp"]["$lte"] = end_timestamp
        
        # Get detections from database
        detections = await database.get_detections(query, limit)
        
        if not detections:
            detections = []
        
        return detections
        
    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register", response_model=Dict[str, Any])
async def register_person(
    registration: RegistrationRequest,
    api_key: bool = Depends(get_api_key),
    face_matcher = Depends(get_face_matcher),
    face_detector = Depends(get_face_detector)
):
    """Register a new person in the system"""
    try:
        if registration.image_base64:
            # Decode base64 image
            try:
                image_data = base64.b64decode(registration.image_base64)
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid image data"}
                    )
            except Exception as e:
                logger.error(f"Error decoding image: {e}")
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Error decoding image: {str(e)}"}
                )
            
            # Detect face in image
            face_detections = face_detector.detect_faces(image)
            
            if not face_detections:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "No face detected in the provided image"}
                )
            
            # Use the first detected face
            face_detection = face_detections[0]
            
            # Add person to database
            success = face_matcher.add_person(
                image,
                face_detection,
                registration.person_data.id,
                registration.person_data.dict()
            )
            
            if not success:
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Failed to add person to database"}
                )
            
            return {
                "status": "success",
                "message": f"Person {registration.person_data.id} registered successfully",
                "person_id": registration.person_data.id
            }
        
        else:
            return JSONResponse(
                status_code=400,
                content={"detail": "Image data is required for registration"}
            )
            
    except Exception as e:
        logger.error(f"Error registering person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload_profile", response_model=Dict[str, Any])
async def upload_profile(
    person_id: str = Form(...),
    name: Optional[str] = Form(None),
    role: Optional[str] = Form(None),
    organization: Optional[str] = Form(None),
    image: UploadFile = File(...),
    api_key: bool = Depends(get_api_key),
    face_matcher = Depends(get_face_matcher),
    face_detector = Depends(get_face_detector)
):
    """Upload a profile image and register a person"""
    try:
        # Read image data
        image_data = await image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid image data"}
            )
        
        # Detect face in image
        face_detections = face_detector.detect_faces(image)
        
        if not face_detections:
            return JSONResponse(
                status_code=400,
                content={"detail": "No face detected in the provided image"}
            )
        
        # Use the first detected face
        face_detection = face_detections[0]
        
        # Prepare person data
        person_data = {
            "id": person_id,
            "registration_time": datetime.utcnow().isoformat()
        }
        
        if name:
            person_data["name"] = name
        
        if role:
            person_data["role"] = role
        
        if organization:
            person_data["organization"] = organization
        
        # Add person to database
        success = face_matcher.add_person(
            image,
            face_detection,
            person_id,
            person_data
        )
        
        if not success:
            return JSONResponse(
                status_code=500,
                content={"detail": "Failed to add person to database"}
            )
        
        return {
            "status": "success",
            "message": f"Person {person_id} registered successfully",
            "person_id": person_id
        }
        
    except Exception as e:
        logger.error(f"Error uploading profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/identities/{identity_id}", response_model=PersonData)
async def get_identity(
    identity_id: str = PathParam(..., description="Identity ID to retrieve"),
    api_key: bool = Depends(get_api_key),
    face_matcher = Depends(get_face_matcher)
):
    """Get identity information by ID"""
    try:
        # Check if identity exists
        if identity_id not in face_matcher.known_identities:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Identity {identity_id} not found"}
            )
        
        # Return identity data
        return face_matcher.known_identities[identity_id]
        
    except Exception as e:
        logger.error(f"Error getting identity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/identities", response_model=List[PersonData])
async def get_identities(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of identities to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    api_key: bool = Depends(get_api_key),
    face_matcher = Depends(get_face_matcher)
):
    """Get all identities"""
    try:
        # Get all identities
        identities = list(face_matcher.known_identities.values())
        
        # Apply pagination
        paginated = identities[offset:offset+limit]
        
        return paginated
        
    except Exception as e:
        logger.error(f"Error getting identities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/identities/{identity_id}", response_model=Dict[str, Any])
async def delete_identity(
    identity_id: str = PathParam(..., description="Identity ID to delete"),
    api_key: bool = Depends(get_api_key),
    face_matcher = Depends(get_face_matcher)
):
    """Delete an identity from the system"""
    try:
        # Check if identity exists
        if identity_id not in face_matcher.known_identities:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Identity {identity_id} not found"}
            )
        
        # Remove from embeddings
        if identity_id in face_matcher.known_embeddings:
            del face_matcher.known_embeddings[identity_id]
        
        # Remove from identities
        del face_matcher.known_identities[identity_id]
        
        # Save database
        face_matcher.save_database()
        
        return {
            "status": "success",
            "message": f"Identity {identity_id} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting identity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cameras", response_model=Dict[str, Any])
async def get_cameras(
    api_key: bool = Depends(get_api_key),
    camera_manager = Depends(get_camera_manager)
):
    """Get list of available cameras"""
    try:
        cameras = {}
        
        for camera_id, camera in camera_manager.cameras.items():
            cameras[camera_id] = {
                "id": camera_id,
                "source": camera.source,
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps,
                "running": camera.is_running,
                "frame_count": camera.frame_count
            }
        
        return {
            "cameras": cameras,
            "count": len(cameras)
        }
        
    except Exception as e:
        logger.error(f"Error getting cameras: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras", response_model=Dict[str, Any])
async def add_camera(
    camera: CameraInfo,
    api_key: bool = Depends(get_api_key),
    camera_manager = Depends(get_camera_manager),
    config: Config = Depends(get_config)
):
    """Add a new camera to the system"""
    try:
        # Check if camera ID already exists
        if camera.id in camera_manager.cameras:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Camera {camera.id} already exists"}
            )
        
        # Create camera instance
        new_camera = Camera(
            camera_id=camera.id,
            source=camera.source,
            width=camera.width,
            height=camera.height,
            fps=camera.fps
        )
        
        # Add to camera manager
        camera_manager.cameras[camera.id] = new_camera
        
        # Start camera if requested
        if camera.running:
            new_camera.start()
            # Start processing task
            task = asyncio.create_task(camera_manager._process_camera(camera.id))
            camera_manager.processing_tasks[camera.id] = task
        
        # Update config
        current_cameras = config.get("cameras", [])
        current_cameras.append({
            "id": camera.id,
            "name": camera.name,
            "source": camera.source,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
            "processing_fps": 15,  # Default processing FPS
            "enabled": True
        })
        
        # Update the entire configuration
        config_data = config.get_all()
        config_data["cameras"] = current_cameras
        config.save_to_file()
        
        return {
            "status": "success",
            "message": f"Camera {camera.id} added successfully",
            "camera": {
                "id": camera.id,
                "source": camera.source,
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps,
                "running": camera.running
            }
        }
        
    except Exception as e:
        logger.error(f"Error adding camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/cameras/{camera_id}", response_model=Dict[str, Any])
async def update_camera(
    camera_id: str = PathParam(..., description="Camera ID to update"),
    camera: CameraInfo = Body(...),
    api_key: bool = Depends(get_api_key),
    camera_manager = Depends(get_camera_manager),
    config: Config = Depends(get_config)
):
    """Update camera settings"""
    try:
        # Check if camera exists
        if camera_id not in camera_manager.cameras:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Camera {camera_id} not found"}
            )
        
        # Get existing camera
        existing_camera = camera_manager.cameras[camera_id]
        
        # Stop camera if running
        was_running = existing_camera.is_running
        if was_running:
            existing_camera.stop()
            if camera_id in camera_manager.processing_tasks:
                task = camera_manager.processing_tasks[camera_id]
                if not task.done():
                    task.cancel()
                await task
        
        # Update camera settings
        new_camera = Camera(
            camera_id=camera_id,
            source=camera.source,
            width=camera.width,
            height=camera.height,
            fps=camera.fps
        )
        
        # Replace in camera manager
        camera_manager.cameras[camera_id] = new_camera
        
        # Start camera if it was running or requested
        if was_running or camera.running:
            new_camera.start()
            # Start processing task
            task = asyncio.create_task(camera_manager._process_camera(camera_id))
            camera_manager.processing_tasks[camera_id] = task
        
        # Update config
        cameras_config = config.get("cameras", [])
        updated = False
        for i, cam_config in enumerate(cameras_config):
            if cam_config.get("id") == camera_id:
                cameras_config[i] = {
                    "id": camera_id,
                    "name": camera.name,
                    "source": camera.source,
                    "width": camera.width,
                    "height": camera.height,
                    "fps": camera.fps,
                    "processing_fps": cam_config.get("processing_fps", 15),  # Preserve existing processing FPS
                    "enabled": True
                }
                updated = True
                break
        
        if not updated:
            cameras_config.append({
                "id": camera_id,
                "name": camera.name,
                "source": camera.source,
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps,
                "processing_fps": 15,  # Default processing FPS
                "enabled": True
            })
        
        # Update the entire configuration
        config_data = config.get_all()
        config_data["cameras"] = cameras_config
        config.save_to_file()
        
        return {
            "status": "success",
            "message": f"Camera {camera_id} updated successfully",
            "camera": {
                "id": camera_id,
                "source": camera.source,
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps,
                "running": new_camera.is_running
            }
        }
        
    except Exception as e:
        logger.error(f"Error updating camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cameras/{camera_id}", response_model=Dict[str, Any])
async def delete_camera(
    camera_id: str = PathParam(..., description="Camera ID to delete"),
    api_key: bool = Depends(get_api_key),
    camera_manager = Depends(get_camera_manager),
    config: Config = Depends(get_config)
):
    """Delete a camera from the system"""
    try:
        # Check if camera exists
        if camera_id not in camera_manager.cameras:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Camera {camera_id} not found"}
            )
        
        # Get camera
        camera = camera_manager.cameras[camera_id]
        
        # Stop camera if running
        if camera.is_running:
            camera.stop()
            if camera_id in camera_manager.processing_tasks:
                task = camera_manager.processing_tasks[camera_id]
                if not task.done():
                    task.cancel()
                await task
        
        # Remove from camera manager
        del camera_manager.cameras[camera_id]
        if camera_id in camera_manager.processing_tasks:
            del camera_manager.processing_tasks[camera_id]
        
        # Update config
        cameras_config = config.get("cameras", [])
        cameras_config = [cam for cam in cameras_config if cam.get("id") != camera_id]
        config.set("cameras", cameras_config)
        config.save_to_file()
        
        return {
            "status": "success",
            "message": f"Camera {camera_id} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cameras/{camera_id}/snapshot", response_class=StreamingResponse)
async def get_camera_snapshot(
    camera_id: str = PathParam(..., description="Camera ID to get snapshot from"),
    api_key: bool = Depends(get_api_key),
    camera_manager = Depends(get_camera_manager)
):
    """Get a snapshot from a camera"""
    try:
        # Check if camera exists
        if camera_id not in camera_manager.cameras:
            raise HTTPException(
                status_code=404,
                detail=f"Camera {camera_id} not found"
            )
        
        # Get camera
        camera = camera_manager.cameras[camera_id]
        
        # Check if camera is running
        if not camera.is_running:
            raise HTTPException(
                status_code=400,
                detail=f"Camera {camera_id} is not running"
            )
        
        # Get latest frame
        frame, timestamp = camera.get_frame()
        
        if frame is None:
            raise HTTPException(
                status_code=404,
                detail=f"No frame available from camera {camera_id}"
            )
        
        # Encode frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        
        # Return as image response
        return StreamingResponse(
            BytesIO(jpeg.tobytes()),
            media_type="image/jpeg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras/{camera_id}/start", response_model=Dict[str, Any])
async def start_camera(
    camera_id: str = PathParam(..., description="Camera ID to start"),
    api_key: bool = Depends(get_api_key),
    camera_manager = Depends(get_camera_manager)
):
    """Start a camera"""
    try:
        # Check if camera exists
        if camera_id not in camera_manager.cameras:
            raise HTTPException(
                status_code=404,
                detail=f"Camera {camera_id} not found"
            )
        
        # Get camera
        camera = camera_manager.cameras[camera_id]
        
        # Start camera if not running
        if not camera.is_running:
            camera.start()
            # Start processing task
            task = asyncio.create_task(camera_manager._process_camera(camera_id))
            camera_manager.processing_tasks[camera_id] = task
        
        return {
            "status": "success",
            "message": f"Camera {camera_id} started successfully",
            "camera": {
                "id": camera_id,
                "source": camera.source,
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps,
                "running": camera.is_running
            }
        }
        
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras/{camera_id}/stop", response_model=Dict[str, Any])
async def stop_camera(
    camera_id: str = PathParam(..., description="Camera ID to stop"),
    api_key: bool = Depends(get_api_key),
    camera_manager = Depends(get_camera_manager)
):
    """Stop a camera"""
    try:
        # Check if camera exists
        if camera_id not in camera_manager.cameras:
            raise HTTPException(
                status_code=404,
                detail=f"Camera {camera_id} not found"
            )
        
        # Get camera
        camera = camera_manager.cameras[camera_id]
        
        # Stop camera if running
        if camera.is_running:
            camera.stop()
            if camera_id in camera_manager.processing_tasks:
                task = camera_manager.processing_tasks[camera_id]
                if not task.done():
                    task.cancel()
                await task
        
        return {
            "status": "success",
            "message": f"Camera {camera_id} stopped successfully",
            "camera": {
                "id": camera_id,
                "source": camera.source,
                "width": camera.width,
                "height": camera.height,
                "fps": camera.fps,
                "running": camera.is_running
            }
        }
        
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export", response_model=Dict[str, Any])
async def export_data(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    api_key: bool = Depends(get_api_key),
    database = Depends(get_database),
    config: Config = Depends(get_config)
):
    """Export data from the system"""
    try:
        # Validate export type
        valid_types = ["detections", "profiles", "analytics"]
        if request.data_type not in valid_types:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Invalid data_type. Must be one of {valid_types}"}
            )
        
        # Validate format
        valid_formats = ["json", "csv"]
        if request.format not in valid_formats:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Invalid format. Must be one of {valid_formats}"}
            )
        
        # Prepare export file path
        export_dir = Path(config.get("storage.export_path", "exports"))
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{request.data_type}_{timestamp}.{request.format}"
        export_path = export_dir / filename
        
        # Prepare query
        query = {}
        
        if request.camera_id:
            query["camera_id"] = request.camera_id
        
        if request.start_date or request.end_date:
            query["timestamp"] = {}
            
            if request.start_date:
                query["timestamp"]["$gte"] = request.start_date
            
            if request.end_date:
                query["timestamp"]["$lte"] = request.end_date
        
        # Export data in background
        background_tasks.add_task(
            export_data_task,
            database,
            request.data_type,
            request.format,
            export_path,
            query
        )
        
        return {
            "status": "success",
            "message": f"Export started. File will be available at {filename}",
            "filename": filename
        }
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/{filename}", response_class=FileResponse)
async def download_export(
    filename: str = PathParam(..., description="Export file to download"),
    api_key: bool = Depends(get_api_key),
    config: Config = Depends(get_config)
):
    """Download an exported file"""
    try:
        # Prepare export file path
        export_dir = Path(config.get("storage.export_path", "exports"))
        export_path = export_dir / filename
        
        # Check if file exists
        if not export_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Export file {filename} not found"
            )
        
        # Return file
        return FileResponse(
            export_path,
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading export: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/maintenance/cleanup", response_model=Dict[str, Any])
async def cleanup_old_data(
    days: int = Query(None, description="Days to keep data (overrides config setting)"),
    api_key: bool = Depends(get_api_key),
    database = Depends(get_database),
    config: Config = Depends(get_config)
):
    """Clean up old data based on retention policy"""
    try:
        # Get retention days from config if not provided
        retention_days = days
        if retention_days is None:
            retention_days = config.get("privacy.data_retention.detection_data_days", 30)
        
        # Calculate cutoff date
        cutoff_date = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
        
        # Delete old data
        deleted_count = await database.cleanup_old_data(cutoff_date)
        
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} records older than {cutoff_date}",
            "deleted_count": deleted_count,
            "cutoff_date": cutoff_date
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up old data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stats/reset")
async def reset_statistics(
    camera_id: Optional[str] = None,
    reset_type: str = Query("all", enum=["all", "performance", "errors", "camera"]),
    older_than_days: Optional[int] = None
) -> Dict[str, Any]:
    """Reset statistics based on parameters"""
    try:
        if reset_type == "all":
            success = db.reset_all_stats()
        elif reset_type == "performance":
            success = db.reset_performance_metrics(camera_id, older_than_days)
        elif reset_type == "errors":
            success = db.reset_error_logs(camera_id)
        elif reset_type == "camera" and camera_id:
            success = db.reset_camera_stats(camera_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid reset type or missing camera_id")
        
        if success:
            # Get updated statistics summary
            summary = db.get_stats_summary(camera_id)
            return {
                "status": "success",
                "message": f"Reset {reset_type} statistics successfully",
                "summary": summary
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reset statistics")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/summary")
async def get_statistics_summary(camera_id: Optional[str] = None) -> Dict[str, Any]:
    """Get summary of current statistics"""
    try:
        summary = db.get_stats_summary(camera_id)
        return {
            "status": "success",
            "data": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======== Utility Functions ========

async def export_data_task(database, data_type, format, export_path, query):
    """Background task to export data"""
    try:
        # Get data based on type
        data = []
        
        if data_type == "detections":
            data = await database.get_detections(query, limit=None)
        elif data_type == "profiles":
            data = await database.get_all_profiles()
        elif data_type == "analytics":
            data = await database.get_analytics(
                start_time=query.get("timestamp", {}).get("$gte"),
                end_time=query.get("timestamp", {}).get("$lte")
            )
        
        # Export based on format
        if format == "json":
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == "csv":
            import csv
            
            # Get field names from first record
            if not data:
                # Create empty file
                with open(export_path, 'w', newline='') as f:
                    f.write("")
                return
            
            # Flatten nested data
            flattened_data = []
            
            for record in data:
                flat_record = {}
                
                # Process each field
                for key, value in record.items():
                    if isinstance(value, dict):
                        # Flatten nested dict
                        for nested_key, nested_value in value.items():
                            flat_record[f"{key}_{nested_key}"] = nested_value
                    elif isinstance(value, list):
                        # Convert list to string
                        flat_record[key] = json.dumps(value)
                    else:
                        flat_record[key] = value
                
                flattened_data.append(flat_record)
            
            # Write CSV
            fieldnames = flattened_data[0].keys()
            
            with open(export_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
        
        logger.info(f"Exported {len(data)} records to {export_path}")
        
    except Exception as e:
        logger.error(f"Error in export task: {e}")
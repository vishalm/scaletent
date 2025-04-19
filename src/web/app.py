"""
Web application for ScaleTent
"""

import asyncio
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
import logging

from src.core.logger import setup_logger
from src.api.routes import router as api_router
from src.core.config import Config
from src.utils.ssl_utils import configure_ssl

logger = setup_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title="ScaleTent API",
    description="API for ScaleTent Meet & Greet System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get base directory
base_dir = Path(__file__).parent

# Set up static files
app.mount(
    "/static",
    StaticFiles(directory=str(base_dir / "static")),
    name="static"
)

# Set up templates
templates = Jinja2Templates(directory=str(base_dir / "templates"))

# Include API routes under /api prefix
app.include_router(api_router, prefix="/api")

# Mock data for development
MOCK_DATA = {
    "status": {
        "total_visitors": 150,
        "known_faces": 75,
        "active_cameras": 3,
        "processing_fps": 25.5
    },
    "detections": {
        "latest": [
            {
                "timestamp": "2024-03-14T12:00:00Z",
                "camera_id": "cam1",
                "person_data": {"name": "John Doe"}
            }
        ]
    },
    "cameras": {
        "cam1": {"id": "cam1", "name": "Main Entrance", "running": True},
        "cam2": {"id": "cam2", "name": "Side Door", "running": False}
    }
}

# API endpoints for development
@app.get("/api/status")
async def get_status():
    """Get system status"""
    return MOCK_DATA["status"]

@app.get("/api/detections/latest")
async def get_latest_detections():
    """Get latest detections"""
    return MOCK_DATA["detections"]

@app.get("/api/cameras")
async def get_cameras():
    """Get camera list"""
    return {"cameras": MOCK_DATA["cameras"]}

# Web routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "ScaleTent Dashboard"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/cameras", response_class=HTMLResponse)
async def cameras(request: Request):
    """Cameras page"""
    return templates.TemplateResponse(
        "cameras.html",
        {
            "request": request,
            "title": "Camera Feeds"
        }
    )

@app.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request):
    """Analytics page"""
    return templates.TemplateResponse(
        "analytics.html",
        {
            "request": request,
            "title": "Detection Analytics"
        }
    )

@app.get("/profiles", response_class=HTMLResponse)
async def profiles(request: Request):
    """Profiles management page"""
    return templates.TemplateResponse(
        "profiles.html",
        {
            "request": request,
            "title": "Profile Management"
        }
    )

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    """Settings page"""
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "title": "System Settings"
        }
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500
        }
    )

if __name__ == "__main__":
    # Configure SSL if needed
    if not configure_ssl():
        logger.warning("Failed to configure SSL certificates")
    
    # Load configuration from environment
    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", 8000))
    reload = os.getenv("WEB_RELOAD", "true").lower() == "true"
    
    # Run the application
    uvicorn.run(
        "src.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
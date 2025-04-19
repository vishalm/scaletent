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
    title="ScaleTent Web Interface",
    description="Web Interface for ScaleTent Meet & Greet System",
    version="1.0.0"
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

# Web interface routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/cameras", response_class=HTMLResponse)
async def cameras(request: Request):
    """Serve the cameras page"""
    return templates.TemplateResponse("cameras.html", {"request": request})

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
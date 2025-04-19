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

class WebApp:
    """
    Web application for ScaleTent
    """
    
    def __init__(self, config):
        """
        Initialize web application
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.app = FastAPI(
            title="ScaleTent API",
            description="API for ScaleTent Meet & Greet System",
            version="1.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc",
            openapi_url="/api/openapi.json"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, replace with specific origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Configure SSL certificates
        if not configure_ssl():
            logger.warning("Failed to configure SSL certificates")
        
        self.setup_routes()
        self.server = None
        
        logger.info("Web application initialized")
    
    def setup_routes(self):
        """Set up web application routes"""
        # Get base directory
        base_dir = Path(__file__).parent
        
        # Set up static files
        self.app.mount(
            "/static",
            StaticFiles(directory=base_dir / "static"),
            name="static"
        )
        
        # Set up templates
        self.templates = Jinja2Templates(directory=base_dir / "templates")
        
        # Include API routes under /api prefix
        self.app.include_router(api_router, prefix="/api")
        
        # Set up web routes
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Main dashboard page"""
            return self.templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "title": "ScaleTent Dashboard"
                }
            )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy"}
        
        @self.app.get("/cameras", response_class=HTMLResponse)
        async def cameras(request: Request):
            """Cameras page"""
            return self.templates.TemplateResponse(
                "cameras.html",
                {
                    "request": request,
                    "title": "Camera Feeds"
                }
            )
        
        @self.app.get("/analytics", response_class=HTMLResponse)
        async def analytics(request: Request):
            """Analytics page"""
            return self.templates.TemplateResponse(
                "analytics.html",
                {
                    "request": request,
                    "title": "Detection Analytics"
                }
            )
        
        @self.app.get("/profiles", response_class=HTMLResponse)
        async def profiles(request: Request):
            """Profiles management page"""
            return self.templates.TemplateResponse(
                "profiles.html",
                {
                    "request": request,
                    "title": "Profile Management"
                }
            )
        
        @self.app.get("/settings", response_class=HTMLResponse)
        async def settings(request: Request):
            """Settings page"""
            return self.templates.TemplateResponse(
                "settings.html",
                {
                    "request": request,
                    "title": "System Settings"
                }
            )
        
        # Error handlers
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions"""
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.detail,
                    "status_code": exc.status_code
                }
            )
        
        @self.app.exception_handler(Exception)
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
    
    async def start(self):
        """Start the web application server"""
        try:
            host = self.config.get("web.host", "0.0.0.0")
            port = self.config.get("web.port", 8000)
            
            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="info",
                ssl_keyfile=self.config.get("ssl.keyfile"),
                ssl_certfile=self.config.get("ssl.certfile"),
                reload=self.config.get("web.reload", False)
            )
            
            self.server = uvicorn.Server(config)
            
            logger.info(f"Starting web application at http{'s' if config.ssl_keyfile else ''}://{host}:{port}")
            
            # Run in a task to avoid blocking
            server_task = asyncio.create_task(self.server.serve())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start web application: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the web application server"""
        if self.server:
            logger.info("Shutting down web application")
            self.server.should_exit = True
            await self.server.shutdown()


def create_app(config: Config) -> FastAPI:
    """Create and configure the FastAPI application."""
    web_app = WebApp(config)
    return web_app.app


if __name__ == "__main__":
    # Load configuration from environment or file
    config_data = {
        "web": {
            "host": os.getenv("WEB_HOST", "localhost"),
            "port": int(os.getenv("WEB_PORT", 8000)),
            "reload": os.getenv("WEB_RELOAD", "false").lower() == "true"
        },
        "ssl": {
            "keyfile": os.getenv("SSL_KEYFILE"),
            "certfile": os.getenv("SSL_CERTFILE")
        }
    }
    config = Config(**config_data)
    app = create_app(config)
    
    uvicorn.run(
        app,
        host=config.get("web.host"),
        port=config.get("web.port"),
        ssl_keyfile=config.get("ssl.keyfile"),
        ssl_certfile=config.get("ssl.certfile"),
        reload=config.get("web.reload", False)
    )
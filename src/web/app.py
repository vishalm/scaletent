"""
Web application for ScaleTent
"""

import asyncio
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn

from core.logger import setup_logger
from api.routes import router as api_router

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
            version="1.0.0"
        )
        
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
        
        # Include API routes
        self.app.include_router(api_router)
        
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
        
        # Default 404 handler
        @self.app.exception_handler(404)
        async def not_found_handler(request: Request, exc: HTTPException):
            """Custom 404 handler"""
            return self.templates.TemplateResponse(
                "404.html",
                {
                    "request": request,
                    "title": "Page Not Found"
                },
                status_code=404
            )
        
        # Default 500 handler
        @self.app.exception_handler(500)
        async def server_error_handler(request: Request, exc: HTTPException):
            """Custom 500 handler"""
            return self.templates.TemplateResponse(
                "500.html",
                {
                    "request": request,
                    "title": "Server Error"
                },
                status_code=500
            )
    
    async def start(self):
        """Start the web application server"""
        try:
            host = self.config.get("web.host", "0.0.0.0")
            port = self.config.get("web.port", 5000)
            
            config = uvicorn.Config(
                self.app,
                host=host,
                port=port,
                log_level="info"
            )
            
            self.server = uvicorn.Server(config)
            
            logger.info(f"Starting web application at http://{host}:{port}")
            
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


def create_app(config):
    """
    Create and configure web application
    
    Args:
        config: Configuration object
    
    Returns:
        WebApp: Web application instance
    """
    return WebApp(config)


if __name__ == "__main__":
    # Run app directly for testing
    from core.config import Config
    
    config_data = {
        "web": {
            "host": "127.0.0.1",
            "port": 5000
        }
    }
    
    config = Config(config_data)
    app = create_app(config)
    
    import uvicorn
    uvicorn.run(
        app.app,
        host=config.get("web.host"),
        port=config.get("web.port")
    )
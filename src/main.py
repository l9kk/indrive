"""FastAPI application for destination prediction service."""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .routers.prediction import router as prediction_router


def create_app() -> FastAPI:
    """Create FastAPI application with configuration."""
    
    app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure as needed for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(prediction_router)
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc)
            }
        )
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup."""
        try:
            # Import here to trigger model loading
            from .services.model_service import get_model_service
            model_service = get_model_service()
            health = model_service.health_check()
            
            if health["status"] != "healthy":
                print(f"Warning: Model service is not healthy: {health}")
            else:
                print(f"Model service initialized successfully: {health['model_type']}")
                
        except Exception as e:
            print(f"Warning: Failed to initialize model service: {e}")
            print("The API will start but predictions may fail until model is trained")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Destination Prediction API",
            "version": settings.API_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    
    return app


# Create app instance
app = create_app()


def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
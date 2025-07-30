from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.database import engine, Base
from app.utils.logging import configure_logging
from app.routers.voice_websocket import router as voice_websocket_router
from app.routers.voice_health import router as voice_health_router
from app.routers import voice_processing

# Import all routers
from app.routers import (
    auth_router,
    agents_router, 
    chat_router,
    rag_router,
    training_router,
    voice_router,
    agent_interaction_router,
    health_router,
    execute_router,
    monitoring_router,
    voice_websocket_router
)

# Import middleware
from app.middleware import LoggingMiddleware, RateLimiterMiddleware
from app.middleware.errorHandlingMiddleware import ErrorHandlingMiddleware
from app.middleware.metrics_middleware import metrics_middleware

import logging

# Configure logging
logger = configure_logging()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI Agent Platform with Multi-Source Internet Search",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)

if settings.RATE_LIMITING_ENABLED:
    app.add_middleware(
        RateLimiterMiddleware,
        max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
        time_window=settings.RATE_LIMIT_TIME_WINDOW
    )

# Add metrics middleware if enabled
app.middleware("http")(metrics_middleware)

# Include routers with proper prefixes
app.include_router(auth_router, prefix=settings.API_PREFIX + "/auth")
app.include_router(agents_router, prefix=settings.API_PREFIX)
app.include_router(chat_router, prefix=settings.API_PREFIX)
app.include_router(rag_router, prefix=settings.API_PREFIX)
app.include_router(training_router, prefix=settings.API_PREFIX + "/training")

# Add the voice WebSocket router with correct prefix
app.include_router(voice_websocket_router, prefix=settings.API_PREFIX)

# Continue with other routers...
app.include_router(agent_interaction_router, prefix=settings.API_PREFIX)
app.include_router(health_router, prefix=settings.API_PREFIX)
app.include_router(execute_router, prefix=settings.API_PREFIX)
app.include_router(monitoring_router, prefix=settings.API_PREFIX)
app.include_router(voice_health_router, prefix=settings.API_PREFIX + "/voice/ws")
app.include_router(voice_processing.router,prefix=settings.API_PREFIX)

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    try:
        # Create database tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("=== AI Agent Platform Started ===")
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        logger.info(f"Debug Mode: {settings.DEBUG}")
        logger.info(f"Internet Search: {'Enabled' if settings.ENABLE_INTERNET_SEARCH else 'Disabled'}")
        if settings.ENABLE_INTERNET_SEARCH:
            logger.info(f"Search Sources: {', '.join(settings.SEARCH_SOURCES)}")
        logger.info("=== Ready to Accept Requests ===")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down AI Agent Platform...")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Agent Platform with Internet Search",
        "version": settings.APP_VERSION,
        "features": {
            "internet_search": settings.ENABLE_INTERNET_SEARCH,
            "search_sources": settings.SEARCH_SOURCES if settings.ENABLE_INTERNET_SEARCH else [],
            "api_docs": "/docs" if settings.DEBUG else "disabled"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": __import__('time').time()
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error occurred",
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

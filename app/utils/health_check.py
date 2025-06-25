from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.config import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health", tags=["Health Check"])
async def health_check():
    """Basic health check endpoint"""
    try:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )
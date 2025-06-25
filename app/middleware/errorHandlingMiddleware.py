from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import traceback
import time
from typing import Callable

logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Let FastAPI handle HTTP exceptions normally
            raise e
            
        except Exception as e:
            # Log the full error with traceback
            process_time = time.time() - start_time
            
            error_details = {
                "method": request.method,
                "url": str(request.url),
                "client": getattr(request.client, "host", "unknown") if request.client else "unknown",
                "process_time": f"{process_time:.3f}s",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            
            logger.error(
                f"Unhandled exception in {request.method} {request.url.path}: {str(e)}",
                extra=error_details
            )
            
            # Return a generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "error_type": type(e).__name__,
                    "timestamp": time.time(),
                    "path": request.url.path
                }
            )
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Log incoming request
        logger.info(f"Incoming request: {request.method} {request.url}")
        
        try:
            response = await call_next(request)
            # Log successful response
            logger.info(
                f"Request completed: {request.method} {request.url} "
                f"- Status: {response.status_code}"
            )
            return response
        except HTTPException as http_exc:
            # Log HTTP exceptions
            logger.warning(
                f"HTTP Exception: {request.method} {request.url} "
                f"- Status: {http_exc.status_code} - Detail: {http_exc.detail}"
            )
            raise
        except Exception as exc:
            # Log unexpected errors
            logger.error(
                f"Unexpected error: {request.method} {request.url} "
                f"- Error: {str(exc)}",
                exc_info=True
            )
            raise
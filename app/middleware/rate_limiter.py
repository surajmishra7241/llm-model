# middleware/rate_limiter.py

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests=1000, time_window=60):
        super().__init__(app)
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_counts = defaultdict(lambda: {'count': 0, 'window_start': time.time()})
        self.lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        
        async with self.lock:
            current_time = time.time()
            record = self.request_counts[client_ip]
            
            # Reset counter if time window has passed
            if current_time - record['window_start'] > self.time_window:
                record['count'] = 0
                record['window_start'] = current_time
            
            # Check rate limit
            if record['count'] >= self.max_requests:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests"},
                    headers={"Retry-After": str(self.time_window)}
                )
            
            record['count'] += 1
        
        response = await call_next(request)
        return response
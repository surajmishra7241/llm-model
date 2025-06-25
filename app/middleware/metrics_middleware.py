from prometheus_client import Counter, Histogram, make_asgi_app
from fastapi import Request, Response
from app.config import settings
import time

if settings.PROMETHEUS_ENABLED:
    REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP Requests",
        ["method", "path", "status_code"]
    )
    
    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP Request Latency",
        ["method", "path"]
    )
    
    metrics_app = make_asgi_app()
    
    async def metrics_middleware(request: Request, call_next):
        if request.url.path == "/metrics":
            return await metrics_app(request.scope, request.receive, request.send)
            
        start_time = time.time()
        method = request.method
        path = request.url.path
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_COUNT.labels(method, path, status_code).inc()
            REQUEST_LATENCY.labels(method, path).observe(duration)
            
        return response
else:
    async def metrics_middleware(request: Request, call_next):
        return await call_next(request)
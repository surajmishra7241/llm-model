from prometheus_client import Counter, Histogram, make_asgi_app, REGISTRY
from fastapi import Request, Response
from app.config import settings
import time

# Only initialize metrics once
_metrics_initialized = False
REQUEST_COUNT = None
REQUEST_LATENCY = None
metrics_app = None

def initialize_metrics():
    global _metrics_initialized, REQUEST_COUNT, REQUEST_LATENCY, metrics_app
    
    if _metrics_initialized or not settings.PROMETHEUS_ENABLED:
        return
    
    try:
        # Clear existing metrics to avoid conflicts
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass
        
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
        _metrics_initialized = True
        
    except Exception as e:
        print(f"Failed to initialize metrics: {e}")

# Initialize metrics
initialize_metrics()

async def metrics_middleware(request: Request, call_next):
    if request.url.path == "/metrics" and metrics_app:
        return await metrics_app(request.scope, request.receive, request.send)
        
    if not settings.PROMETHEUS_ENABLED or not REQUEST_COUNT:
        return await call_next(request)
        
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

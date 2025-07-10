# ./app/utils/monitoring.py
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from prometheus_client import generate_latest, REGISTRY
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP Request Latency',
    ['method', 'endpoint']
)

AGENT_EXECUTIONS = Counter(
    'agent_executions_total',
    'Total Agent Executions',
    ['agent_id', 'status']
)

RAG_REQUESTS = Counter(
    'rag_requests_total',
    'Total RAG Requests',
    ['status']
)

async def initialize():
    if settings.PROMETHEUS_ENABLED:
        start_http_server(settings.PROMETHEUS_PORT)
        logger.info(f"Metrics server started on port {settings.PROMETHEUS_PORT}")

def get_metrics():
    return generate_latest(REGISTRY)

def record_request(method: str, endpoint: str, status_code: int, duration: float):
    REQUEST_COUNT.labels(method, endpoint, status_code).inc()
    REQUEST_LATENCY.labels(method, endpoint).observe(duration)

def record_agent_execution(agent_id: str, success: bool):
    status = "success" if success else "failure"
    AGENT_EXECUTIONS.labels(agent_id, status).inc()

def record_rag_request(success: bool):
    status = "success" if success else "failure"
    RAG_REQUESTS.labels(status).inc()
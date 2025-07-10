# ./app/routers/monitoring.py
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.utils.monitoring import get_metrics
from app.dependencies import get_current_user

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

@router.get("/metrics")
async def metrics_endpoint(user: dict = Depends(get_current_user)):
    """Get Prometheus metrics"""
    if not user.get("is_admin", False):
        return JSONResponse(
            status_code=403,
            content={"detail": "Only admin users can access metrics"}
        )
    return get_metrics()
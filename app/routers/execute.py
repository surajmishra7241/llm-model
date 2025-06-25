from fastapi import APIRouter, Depends
from app.dependencies import get_current_user
from app.services.agent_service import execute_agent

router = APIRouter(prefix="/execute", tags=["Execution"])

@router.post("/{agent_id}")
async def execute_agent_endpoint(
    agent_id: str,
    input_data: dict,
    user: dict = Depends(get_current_user)
):
    return await execute_agent(agent_id, user["sub"], input_data)
from fastapi import APIRouter, Depends, HTTPException
from app.dependencies import get_current_user, get_db
from app.services.agent_execution_service import execute_agent
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/execute", tags=["Execution"])

@router.post("/{agent_id}")
async def execute_agent_endpoint(
    agent_id: str,
    request_data: dict,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        # The execute_agent service expects input_data with "input" key
        input_data = {
            "input": request_data.get("input", ""),
            "parameters": request_data.get("parameters", {})
        }
        
        result = await execute_agent(agent_id, user["sub"], input_data, db)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

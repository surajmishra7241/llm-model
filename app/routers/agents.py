# Updated agents.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.agent_model import AgentCreate, AgentUpdate, AgentResponse
from app.services.agent_service import AgentService
from app.dependencies import get_db, get_current_user
from typing import List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.models.agent_model import AgentCreate, AgentUpdate, AgentResponse


router = APIRouter(prefix="/agents", tags=["agents"])

@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_data: "AgentCreate",
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    service = AgentService(db)
    try:
        return await service.create_agent(user["sub"], agent_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("", response_model=List[AgentResponse])
async def list_agents(
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    service = AgentService(db)
    return await service.list_agents(user["sub"])

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    service = AgentService(db)
    agent = await service.get_agent(agent_id, user["sub"])
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    agent_data: AgentUpdate,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    service = AgentService(db)
    agent = await service.update_agent(agent_id, user["sub"], agent_data)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent

@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    service = AgentService(db)
    if not await service.delete_agent(agent_id, user["sub"]):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
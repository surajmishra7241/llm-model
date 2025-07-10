from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.agent_model import AgentCreate, AgentUpdate, AgentResponse
from app.services.agent_service import AgentService
from app.dependencies import get_db, get_current_user
from typing import List
from sqlalchemy import select, delete
from app.models.db_models import DBAgent
from typing import TYPE_CHECKING

router = APIRouter(prefix="/agents", tags=["agents"])
if TYPE_CHECKING:
    from app.models.agent_model import AgentCreate, AgentUpdate, AgentResponse


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_data: AgentCreate,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    db_agent = DBAgent(
        owner_id=user["sub"],
        name=agent_data.name,
        description=agent_data.description,
        model=agent_data.model,
        system_prompt=agent_data.system_prompt,
        is_public=agent_data.is_public,
        tools=agent_data.tools,
        agent_metadata=agent_data.metadata
    )
    db.add(db_agent)
    await db.commit()
    await db.refresh(db_agent)
    return db_agent

@router.get("", response_model=List[AgentResponse])
async def list_agents(
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    result = await db.execute(
        select(DBAgent).where(DBAgent.owner_id == user["sub"])
    )
    return result.scalars().all()

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
    """Delete an agent and all its associated data"""
    # Import services inside the function to avoid dependency validation issues
    from app.services.rag_service import RAGService
    from app.services.conversation_service import ConversationService
    
    # Verify agent belongs to user
    result = await db.execute(
        select(DBAgent).where(
            DBAgent.id == agent_id,
            DBAgent.owner_id == user["sub"]
        )
    )
    agent = result.scalars().first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    
    # Create service instances inside the function
    rag_service = RAGService(db)
    conversation_service = ConversationService(db)
    
    # Delete all agent data
    await rag_service.delete_agent_data(agent_id, user["sub"])
    await conversation_service.clear_conversation(agent_id, user["sub"])
    
    # Delete agent
    await db.execute(
        delete(DBAgent).where(
            DBAgent.id == agent_id,
            DBAgent.owner_id == user["sub"]
        )
    )
    await db.commit()
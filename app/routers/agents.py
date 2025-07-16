from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.agent_model import AgentCreate, AgentUpdate, AgentResponse
from app.services.agent_service import AgentService
from app.dependencies import get_db, get_current_user
from typing import List
from sqlalchemy import select, delete
from app.models.db_models import DBAgent
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field  # Added Field import here

router = APIRouter(prefix="/agents", tags=["agents"])

if TYPE_CHECKING:
    from app.models.agent_model import AgentCreate, AgentUpdate, AgentResponse

class VoiceModelConfig(BaseModel):
    """Configuration for voice models"""
    model_name: str = Field(default="default", description="Voice model to use")
    speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech rate multiplier")
    pitch: float = Field(default=0.0, ge=-20.0, le=20.0, description="Pitch adjustment in semitones")
    volume_gain: float = Field(default=0.0, ge=-96.0, le=16.0, description="Volume gain in dB")

@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_data: AgentCreate,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    # Initialize default voice config if not provided
    if not hasattr(agent_data, 'metadata') or not agent_data.metadata:
        agent_data.metadata = {}
    if "voice_config" not in agent_data.metadata:
        agent_data.metadata["voice_config"] = VoiceModelConfig().dict()
    
    db_agent = DBAgent(
        owner_id=user["sub"],
        name=agent_data.name,
        description=agent_data.description,
        model=agent_data.model,
        system_prompt=agent_data.system_prompt,
        is_public=agent_data.is_public,
        tools=agent_data.tools,
        agent_metadata=agent_data.metadata,
        voice_enabled=True  # Enable voice by default
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
    
    # Ensure voice config exists
    if not agent.agent_metadata or "voice_config" not in agent.agent_metadata:
        if not agent.agent_metadata:
            agent.agent_metadata = {}
        agent.agent_metadata["voice_config"] = VoiceModelConfig().dict()
        await db.commit()
        await db.refresh(agent)
    
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
    
    # Handle voice config updates
    if hasattr(agent_data, 'metadata') and agent_data.metadata and "voice_config" in agent_data.metadata:
        # Validate voice config
        try:
            VoiceModelConfig(**agent_data.metadata["voice_config"])
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid voice configuration: {str(e)}"
            )
        
        # Update voice config
        if not agent.agent_metadata:
            agent.agent_metadata = {}
        agent.agent_metadata["voice_config"] = agent_data.metadata["voice_config"]
        agent.voice_enabled = True
        await db.commit()
        await db.refresh(agent)
    
    return agent

@router.patch("/{agent_id}/voice", response_model=AgentResponse)
async def update_voice_config(
    agent_id: str,
    voice_config: VoiceModelConfig,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    """Update only the voice configuration for an agent"""
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
    
    # Update voice config
    if not agent.agent_metadata:
        agent.agent_metadata = {}
    agent.agent_metadata["voice_config"] = voice_config.dict()
    agent.voice_enabled = True  # Ensure voice stays enabled
    
    await db.commit()
    await db.refresh(agent)
    return agent

@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    """Delete an agent and all its associated data"""
    from app.services.rag_service import RAGService
    from app.services.conversation_service import ConversationService
    
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
    
    rag_service = RAGService(db)
    conversation_service = ConversationService(db)
    
    await rag_service.delete_agent_data(agent_id, user["sub"])
    await conversation_service.clear_conversation(agent_id, user["sub"])
    
    await db.execute(
        delete(DBAgent).where(
            DBAgent.id == agent_id,
            DBAgent.owner_id == user["sub"]
        )
    )
    await db.commit()
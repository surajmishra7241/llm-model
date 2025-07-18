from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.agent_model import AgentCreate, AgentUpdate, AgentResponse, AgentPersonality
from app.services.agent_service import AgentService
from app.dependencies import get_db, get_current_user
from typing import List
from sqlalchemy import select, delete
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field
from datetime import datetime
import logging
from app.services.conversation_service import ConversationService
from app.services.rag_service import RAGService
from app.models.db_models import (
    DBAgent, 
    DBConversation, 
    DBMessage, 
    DBDocument, 
    DBTrainingJob
)

logger = logging.getLogger(__name__)

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
    try:
        # Initialize default voice config if not provided
        # Fix: Use agent_metadata instead of metadata
        metadata = agent_data.agent_metadata or {}
        
        if "voice_config" not in metadata:
            metadata["voice_config"] = VoiceModelConfig().dict()
        
        # Convert personality to dict for storage
        personality_dict = agent_data.personality.dict() if agent_data.personality else {}
        
        db_agent = DBAgent(
            owner_id=user["sub"],
            name=agent_data.name,
            description=agent_data.description,
            model=agent_data.model,
            system_prompt=agent_data.system_prompt,
            is_public=agent_data.is_public,
            tools=agent_data.tools,
            personality=personality_dict,  # Add personality field
            agent_metadata=metadata,
            voice_enabled=True
        )
        
        db.add(db_agent)
        await db.commit()
        await db.refresh(db_agent)
        
        # Create response with proper field mapping
        response_data = {
            "id": str(db_agent.id),
            "owner_id": db_agent.owner_id,
            "name": db_agent.name,
            "description": db_agent.description,
            "model": db_agent.model,
            "system_prompt": db_agent.system_prompt,
            "is_public": db_agent.is_public,
            "tools": db_agent.tools,
            "personality": AgentPersonality(**db_agent.personality) if db_agent.personality else AgentPersonality(),
            "agent_metadata": db_agent.agent_metadata,
            "created_at": db_agent.created_at,
            "updated_at": db_agent.updated_at
        }
        
        return AgentResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Failed to create agent: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create agent: {str(e)}"
        )






@router.get("", response_model=List[AgentResponse])
async def list_agents(
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    result = await db.execute(
        select(DBAgent).where(DBAgent.owner_id == user["sub"])
    )
    agents = result.scalars().all()
    
    # Convert to response format
    response_agents = []
    for agent in agents:
        response_data = {
            "id": str(agent.id),
            "owner_id": agent.owner_id,
            "name": agent.name,
            "description": agent.description,
            "model": agent.model,
            "system_prompt": agent.system_prompt,
            "is_public": agent.is_public,
            "tools": agent.tools,
            "personality": AgentPersonality(**agent.personality) if agent.personality else AgentPersonality(),
            "agent_metadata": agent.agent_metadata,
            "created_at": agent.created_at,
            "updated_at": agent.updated_at
        }
        response_agents.append(AgentResponse(**response_data))
    
    return response_agents


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
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
    
    # Ensure voice config exists
    if not agent.agent_metadata or "voice_config" not in agent.agent_metadata:
        if not agent.agent_metadata:
            agent.agent_metadata = {}
        agent.agent_metadata["voice_config"] = VoiceModelConfig().dict()
        await db.commit()
        await db.refresh(agent)
    
    # Convert to response format
    response_data = {
        "id": str(agent.id),
        "owner_id": agent.owner_id,
        "name": agent.name,
        "description": agent.description,
        "model": agent.model,
        "system_prompt": agent.system_prompt,
        "is_public": agent.is_public,
        "tools": agent.tools,
        "personality": AgentPersonality(**agent.personality) if agent.personality else AgentPersonality(),
        "agent_metadata": agent.agent_metadata,
        "created_at": agent.created_at,
        "updated_at": agent.updated_at
    }
    
    return AgentResponse(**response_data)


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    agent_data: AgentUpdate,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    try:
        # Get existing agent
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
        
        # Update fields that are provided
        update_data = agent_data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if field == "personality" and value is not None:
                # **FIX: Replace personality completely instead of merging**
                if isinstance(value, AgentPersonality):
                    agent.personality = value.dict()
                else:
                    # Handle partial personality updates properly
                    new_personality = AgentPersonality(**value).dict()
                    agent.personality = new_personality
            elif field == "agent_metadata" and value is not None:
                # Merge metadata (this is fine for metadata)
                existing_metadata = agent.agent_metadata or {}
                existing_metadata.update(value)
                agent.agent_metadata = existing_metadata
            elif hasattr(agent, field) and value is not None:
                setattr(agent, field, value)
        
        agent.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(agent)
        
        # Convert to response
        response_data = {
            "id": str(agent.id),
            "owner_id": agent.owner_id,
            "name": agent.name,
            "description": agent.description,
            "model": agent.model,
            "system_prompt": agent.system_prompt,
            "is_public": agent.is_public,
            "tools": agent.tools,
            "personality": AgentPersonality(**agent.personality) if agent.personality else AgentPersonality(),
            "agent_metadata": agent.agent_metadata,
            "created_at": agent.created_at,
            "updated_at": agent.updated_at
        }
        
        return AgentResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update agent: {str(e)}"
        )



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

@router.delete("/{agent_id}", status_code=status.HTTP_200_OK)
async def delete_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    """Delete an agent and all its associated data"""
    try:
        # Get existing agent first
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
        
        # Initialize services correctly
        rag_service = RAGService()  # ✅ FIXED - no db parameter
        await rag_service.initialize()  # Initialize the service
        
        conversation_service = ConversationService(db)  # ConversationService does take db
        
        # Clean up associated data
        try:
            # Delete agent's RAG data
            await rag_service.delete_agent_data(agent_id, user["sub"])
            
            # Clear conversation history
            await conversation_service.clear_conversation(agent_id, user["sub"])
            
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {str(cleanup_error)}")
            # Continue with agent deletion even if cleanup fails
        
        # Delete the agent record

        await db.execute(
            delete(DBConversation).where(
                DBConversation.agent_id == agent_id,
                DBConversation.user_id == user["sub"]
            )
        )

# Then delete the agent
        await db.execute(
            delete(DBAgent).where(
                DBAgent.id == agent_id,
                DBAgent.owner_id == user["sub"]
            )
        )
        await db.commit()
        
        logger.info(f"Successfully deleted agent {agent_id}")
        
        # Clean up RAG service resources
        await rag_service.close()
        
        return {
            "message": "Agent deleted successfully",
            "agent_id": agent_id,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete agent: {str(e)}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent: {str(e)}"
        )

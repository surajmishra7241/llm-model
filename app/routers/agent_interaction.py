# ./app/routers/agent_interaction.py
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from fastapi.responses import Response
from app.services.agent_interaction_service import AgentInteractionService
from app.dependencies import get_db, get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import logging
import base64

router = APIRouter(prefix="/interact", tags=["Agent Interaction"])
logger = logging.getLogger(__name__)

@router.post("/chat/{agent_id}")
async def chat_with_agent(
    agent_id: str,
    input_data: dict,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Chat with an agent using text"""
    service = AgentInteractionService()
    await service.initialize()
    
    try:
        response = await service.process_input(
            agent_id=agent_id,
            user_id=user["sub"],
            input_text=input_data.get("message"),
            db=db
        )
        return response
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/voice/{agent_id}")
async def voice_interaction(
    agent_id: str,
    audio_file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Interact with an agent using voice"""
    service = AgentInteractionService()
    await service.initialize()
    
    try:
        # Validate audio file
        if audio_file.content_type and not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        logger.info(f"Processing voice interaction for agent {agent_id}")
        logger.info(f"Audio file: {audio_file.filename}, size: {audio_file.size if hasattr(audio_file, 'size') else 'unknown'}")
        
        response = await service.process_input(
            agent_id=agent_id,
            user_id=user["sub"],
            audio_file=audio_file,
            db=db
        )
        
        # Convert response to speech
        speech_response = await service.text_to_speech(
            response["text_response"],
            response["emotional_state"]
        )
        
        return {
            "text": response["text_response"],
            "audio": base64.b64encode(speech_response).decode() if speech_response else None,
            "emotion": response["emotional_state"],
            "context_used": response.get("context_used", [])
        }
    except Exception as e:
        logger.error(f"Voice interaction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{agent_id}")
async def get_conversation_history(
    agent_id: str,
    user: dict = Depends(get_current_user)
):
    """Get conversation history for a user with specific agent"""
    service = AgentInteractionService()
    
    try:
        history = await service.get_conversation_history(user["sub"])
        return {"history": history}
    except Exception as e:
        logger.error(f"Failed to get conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history/{agent_id}")
async def clear_conversation_history(
    agent_id: str,
    user: dict = Depends(get_current_user)
):
    """Clear conversation history for a user with specific agent"""
    service = AgentInteractionService()
    
    try:
        await service.clear_memory(user["sub"])
        return {"message": "Conversation history cleared"}
    except Exception as e:
        logger.error(f"Failed to clear conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
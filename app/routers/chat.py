# app/routers/chat.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from typing import Optional

from app.models.response_schema import ChatRequest, ChatResponse
from app.services.llm_service import OllamaService
from app.dependencies import get_current_user, get_db
from app.services.conversation_service import ConversationService
from app.models.db_models import DBAgent
from app.services.cache import cache_service
import json

router = APIRouter(prefix="/agents/{agent_id}", tags=["chat"])

async def get_chat_history_from_cache(agent_id: str, user_id: str) -> list:
    """Get chat history from Redis cache"""
    cache_key = f"chat_history:{user_id}:{agent_id}"
    history = await cache_service.get(cache_key)
    return json.loads(history) if history else []

async def save_chat_history_to_cache(agent_id: str, user_id: str, history: list):
    """Save chat history to Redis cache"""
    cache_key = f"chat_history:{user_id}:{agent_id}"
    await cache_service.set(cache_key, json.dumps(history), ttl=86400)  # 24 hours TTL

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    agent_id: str,
    chat_request: ChatRequest,
    user: dict = Depends(get_current_user),
    llm_service: OllamaService = Depends(),
    db: AsyncSession = Depends(get_db),  # Proper dependency injection
):
    # Verify agent exists and user has access
    try:
        result = await db.execute(
            select(DBAgent).where(
                DBAgent.id == agent_id,
                or_(
                    DBAgent.owner_id == user["sub"],
                    DBAgent.is_public == True
                )
            )
        )
        agent = result.scalars().first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found or access denied"
            )
        
        # Get conversation history from Redis
        conversation_history = await get_chat_history_from_cache(agent_id, user["sub"])
        
        # Build messages with context
        messages = [
            {"role": "system", "content": agent.system_prompt},
            *conversation_history,
            {"role": "user", "content": chat_request.message}
        ]
        
        # Get response from LLM
        response = await llm_service.chat(
            messages=messages,
            model=agent.model,
            options=chat_request.options
        )
        
        # Update conversation history
        new_history = conversation_history + [
            {"role": "user", "content": chat_request.message},
            {"role": "assistant", "content": response.get("message", {}).get("content", "")}
        ]
        
        # Keep only last 10 messages to avoid too large context
        new_history = new_history[-10:]
        
        # Save updated history to Redis
        await save_chat_history_to_cache(agent_id, user["sub"], new_history)
        
        return ChatResponse(
            message=response.get("message", {}).get("content", ""),
            model=agent.model,
            context=new_history,
            tokens_used=response.get("eval_count", 0)
        )
    finally:
        await db.close()
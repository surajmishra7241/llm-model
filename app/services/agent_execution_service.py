# app/services/agent_execution_service.py
from datetime import datetime
import httpx
import json
from app.services.agent_service import AgentService
from app.models.db_models import DBAgent  # ← ADD THIS
from app.models.agent_model import AgentPersonality  # ← ADD THIS
from app.services.cache import cache_service  # ← ADD THIS
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging

logger = logging.getLogger(__name__)

async def call_ai_service(
    model: str,
    system_prompt: str,
    message: str,
    parameters: dict
):
    """Call your AI service (e.g., Ollama) to execute the agent"""
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": model,
                "prompt": f"{system_prompt}\n\nUser: {message}",
                "stream": False,
                **parameters
            }
            
            response = await client.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "No response generated")
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling AI service: {str(e)}")
        raise Exception(f"AI service returned HTTP error: {str(e)}")
    except httpx.RequestError as e:
        logger.error(f"Request error calling AI service: {str(e)}")
        raise Exception(f"Failed to connect to AI service: {str(e)}")
    except Exception as e:
        logger.error(f"Error calling AI service: {str(e)}")
        raise Exception(f"AI service call failed: {str(e)}")

async def execute_agent(agent_id: str, user_id: str, input_data: dict, db: AsyncSession):
    try:
        # Get agent with personality
        result = await db.execute(
            select(DBAgent).where(
                DBAgent.id == agent_id,
                DBAgent.owner_id == user_id
            )
        )
        agent = result.scalars().first()
        
        if not agent:
            raise ValueError("Agent not found")
        
        # Build personality-aware system prompt
        personality = AgentPersonality(**agent.personality) if agent.personality else AgentPersonality()
        
        enhanced_system_prompt = f"""
{agent.system_prompt}

PERSONALITY CONFIGURATION:
- Traits: {', '.join(personality.traits)}
- Tone: {personality.base_tone}
- Empathy Level: {personality.emotional_awareness.empathy_level}/1.0
- Context Memory: {personality.memory.context_window} messages
- Emotional Awareness: {'Enabled' if personality.emotional_awareness.detect_emotion else 'Disabled'}

Remember to embody these personality traits in all your responses.
"""
        
        # Get conversation history
        history_key = f"chat_history:{user_id}:{agent_id}"
        history = await cache_service.get(history_key)
        conversation_history = json.loads(history) if history else []
        
        # Limit history based on personality memory settings
        max_history = personality.memory.context_window
        conversation_history = conversation_history[-max_history:] if conversation_history else []
        
        # Build messages
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            *conversation_history,
            {"role": "user", "content": input_data.get("input", "")}
        ]
        
        # Execute with LLM
        from app.services.llm_service import OllamaService
        llm_service = OllamaService()
        
        response = await llm_service.chat(
            messages=messages,
            model=agent.model,
            options={
                "temperature": input_data.get("parameters", {}).get("temperature", 0.7),
                "max_tokens": input_data.get("parameters", {}).get("max_tokens", 2000),
                "top_p": 0.9,
            }
        )
        
        # Update conversation history
        new_message_user = {"role": "user", "content": input_data.get("input", "")}
        new_message_assistant = {"role": "assistant", "content": response.get("message", {}).get("content", "")}
        
        updated_history = conversation_history + [new_message_user, new_message_assistant]
        updated_history = updated_history[-max_history:]
        
        # Save to cache
        await cache_service.set(
            history_key, 
            json.dumps(updated_history), 
            ttl=personality.memory.memory_refresh_interval
        )
        
        return {
            "response": response.get("message", {}).get("content", ""),
            "model": agent.model,
            "personality_applied": personality.dict(),
            "context_length": len(updated_history)
        }
        
    except Exception as e:
        logger.error(f"Agent execution failed: {str(e)}")
        raise

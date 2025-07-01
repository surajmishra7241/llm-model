# app/services/agent_execution_service.py
from datetime import datetime
import httpx
from app.services.agent_service import AgentService
from sqlalchemy.ext.asyncio import AsyncSession
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
                "http://localhost:11434/api/generate",  # Ollama endpoint
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

async def execute_agent(
    agent_id: str, 
    owner_id: str, 
    input_data: dict,
    db: AsyncSession
):
    """
    Execute an agent with the given input data.
    """
    agent_service = AgentService(db)
    
    try:
        # Get the agent
        agent = await agent_service.get_agent(agent_id, owner_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Extract message and parameters
        message = input_data.get("message", "")
        parameters = input_data.get("parameters", {})
        
        # Execute the agent
        try:
            response = await call_ai_service(
                model=agent.model,
                system_prompt=agent.system_prompt,
                message=message,
                parameters=parameters
            )
            
            return {
                "status": "success",
                "response": response,
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        await db.rollback()
        raise
    finally:
        await db.close()
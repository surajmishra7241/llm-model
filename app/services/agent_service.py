# Updated agent_service.py
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.db_models import DBAgent
from app.models.agent_model import AgentCreate, AgentUpdate
from datetime import datetime
from sqlalchemy import select, update, delete
import httpx
import json
from app.dependencies import get_db_session

class AgentService:
    def __init__(self, db: AsyncSession = Depends(get_db_session)):
        self.db = db

    async def create_agent(self, owner_id: str, agent_data: AgentCreate) -> DBAgent:
        db_agent = DBAgent(
            owner_id=owner_id,
            name=agent_data.name,
            description=agent_data.description,
            model=agent_data.model,
            system_prompt=agent_data.system_prompt,
            is_public=agent_data.is_public,
            tools=agent_data.tools,
            agent_metadata=agent_data.metadata
        )
        self.db.add(db_agent)
        await self.db.commit()
        await self.db.refresh(db_agent)
        return db_agent

    async def get_agent(self, agent_id: str, owner_id: str) -> DBAgent:
        result = await self.db.execute(
            select(DBAgent).where(
                DBAgent.id == agent_id,
                DBAgent.owner_id == owner_id
            )
        )
        return result.scalars().first()

    async def list_agents(self, owner_id: str) -> list[DBAgent]:
        result = await self.db.execute(
            select(DBAgent).where(DBAgent.owner_id == owner_id)
        )
        return result.scalars().all()

    async def update_agent(self, agent_id: str, owner_id: str, update_data: AgentUpdate) -> DBAgent:
        update_dict = update_data.dict(exclude_unset=True)
        if not update_dict:
            return await self.get_agent(agent_id, owner_id)
            
        await self.db.execute(
            update(DBAgent)
            .where(
                DBAgent.id == agent_id,
                DBAgent.owner_id == owner_id
            )
            .values(**update_dict)
        )
        await self.db.commit()
        return await self.get_agent(agent_id, owner_id)

    async def delete_agent(self, agent_id: str, owner_id: str) -> bool:
        result = await self.db.execute(
            delete(DBAgent)
            .where(
                DBAgent.id == agent_id,
                DBAgent.owner_id == owner_id
            )
        )
        await self.db.commit()
        return result.rowcount > 0


# Standalone execute_agent function
async def execute_agent(agent_id: str, owner_id: str, input_data: dict):
    """
    Execute an agent with the given input data.
    This function should be implemented based on your specific agent execution logic.
    """
    from app.database import get_async_session
    
    # Get database session
    async with get_async_session() as db:
        agent_service = AgentService(db)
        
        # Get the agent
        agent = await agent_service.get_agent(agent_id, owner_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Extract message and parameters
        message = input_data.get("message", "")
        parameters = input_data.get("parameters", {})
        
        # Here you would implement your agent execution logic
        # This is a placeholder implementation
        try:
            # Example: Call to Ollama or your AI service
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


async def call_ai_service(model: str, system_prompt: str, message: str, parameters: dict):
    """
    Call your AI service (e.g., Ollama) to execute the agent.
    Replace this with your actual AI service integration.
    """
    # Example implementation for Ollama
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": model,
                "prompt": f"{system_prompt}\n\nUser: {message}",
                "stream": False,
                **parameters
            }
            
            response = await client.post(
                "http://localhost:11434/api/generate",  # Adjust URL as needed
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "No response generated")
            
    except Exception as e:
        raise Exception(f"AI service call failed: {str(e)}")
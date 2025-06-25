# Updated agent_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.db_models import DBAgent
from app.models.agent_model import AgentCreate, AgentUpdate
from datetime import datetime
from sqlalchemy import select, update, delete

class AgentService:
    def __init__(self, db: AsyncSession):
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
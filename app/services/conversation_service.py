import logging
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, delete
from app.models.db_models import DBConversation, DBMessage
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ConversationService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.memory_cache = {}
    
    async def get_conversation(self, agent_id: str, user_id: str) -> DBConversation:
        """Get or create a conversation between user and agent"""
        result = await self.db.execute(
            select(DBConversation)
            .where(
                DBConversation.agent_id == agent_id,
                DBConversation.user_id == user_id
            )
            .order_by(desc(DBConversation.updated_at))
            .limit(1)
        )
        conversation = result.scalars().first()
        
        if not conversation:
            conversation = DBConversation(
                id=str(uuid.uuid4()),
                agent_id=agent_id,
                user_id=user_id,
                memory_id=str(uuid.uuid4()),
                context={"messages": []},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.db.add(conversation)
            await self.db.commit()
            await self.db.refresh(conversation)
        
        return conversation
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> DBMessage:
        """Add a message to the conversation"""
        message = DBMessage(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata or {},
            created_at=datetime.utcnow()
        )
        self.db.add(message)
        await self.db.commit()
        await self.db.refresh(message)
        return message
    
    async def get_conversation_history(
        self,
        user_id: str,
        agent_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """Get conversation history for context"""
        conversation = await self.get_conversation(agent_id, user_id)
        
        result = await self.db.execute(
            select(DBMessage)
            .where(DBMessage.conversation_id == conversation.id)
            .order_by(desc(DBMessage.created_at))
            .limit(limit)
        )
        
        messages = result.scalars().all()
        
        # Format for LLM: [{"role": "user", "content": "..."}, ...]
        history = []
        for msg in reversed(messages):  # Oldest first
            history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at.isoformat()
            })
        
        return history
    
    async def save_interaction(
        self,
        user_id: str,
        agent_id: str,
        user_message: str,
        agent_response: str,
        metadata: Optional[Dict] = None
    ):
        """Save full interaction to database"""
        conversation = await self.get_conversation(agent_id, user_id)
        
        # Save user message
        await self.add_message(
            conversation_id=conversation.id,
            role="user",
            content=user_message,
            metadata=metadata
        )
        
        # Save agent response
        await self.add_message(
            conversation_id=conversation.id,
            role="agent",
            content=agent_response,
            metadata=metadata
        )
        
        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()
        await self.db.commit()
    
    async def clear_conversation(
        self,
        agent_id: str,
        user_id: str
    ) -> bool:
        """Clear conversation history for an agent-user pair"""
        try:
            # Get conversation
            conversation = await self.get_conversation(agent_id, user_id)
            
            # Delete all messages
            await self.db.execute(
                delete(DBMessage)
                .where(DBMessage.conversation_id == conversation.id)
            )
            
            # Reset conversation context
            conversation.context = {"messages": []}
            conversation.updated_at = datetime.utcnow()
            
            await self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation: {str(e)}")
            await self.db.rollback()
            return False
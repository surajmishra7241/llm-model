# app/services/conversation_service.py
import logging
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from app.models.db_models import DBConversation, DBMessage
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ConversationService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.memory_cache = {}
    
    async def get_conversation(self, agent_id: str, user_id: str) -> DBConversation:
        cache_key = f"{agent_id}_{user_id}"
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
            
        result = await self.db.execute(
            select(DBConversation).where(
                DBConversation.agent_id == agent_id,
                DBConversation.user_id == user_id
            ).order_by(DBConversation.updated_at.desc()).limit(1)
        )
        conversation = result.scalars().first()
        
        if not conversation:
            conversation = DBConversation(
                agent_id=agent_id,
                user_id=user_id,
                memory_id=str(uuid.uuid4()),
                context={"messages": []}
            )
            self.db.add(conversation)
            await self.db.commit()
            await self.db.refresh(conversation)
        
        self.memory_cache[cache_key] = conversation
        return conversation
    
    async def add_message(
        self,
        conversation: DBConversation,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> DBMessage:
        message = DBMessage(
            conversation_id=conversation.id,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.db.add(message)
        
        # Update context (simplified - in production you'd want summarization)
        messages = conversation.context.get("messages", [])
        messages.append({"role": role, "content": content})
        if len(messages) > 10:
            messages = messages[-10:]
        
        conversation.context = {"messages": messages}
        conversation.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(message)
        return message
    
    async def clear_conversation(self, agent_id: str, user_id: str) -> bool:
        try:
            # Delete messages
            await self.db.execute(
                delete(DBMessage).where(
                    DBMessage.conversation.has(
                        DBConversation.agent_id == agent_id,
                        DBConversation.user_id == user_id
                    )
                )
            )
            
            # Delete conversation
            await self.db.execute(
                delete(DBConversation).where(
                    DBConversation.agent_id == agent_id,
                    DBConversation.user_id == user_id
                )
            )
            
            await self.db.commit()
            
            # Clear cache
            cache_key = f"{agent_id}_{user_id}"
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
                
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation: {str(e)}")
            await self.db.rollback()
            return False
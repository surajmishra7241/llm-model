import logging
from fastapi import Depends
from app.services.llm_service import OllamaService
from app.services.rag_service import RAGService
from app.services.conversation_service import ConversationService
from app.models.agent_model import Agent
from app.config import settings

logger = logging.getLogger(__name__)

class VoiceAgentService:
    def __init__(
        self,
        llm_service: OllamaService = Depends(),
        rag_service: RAGService = Depends(),
        conversation_service: ConversationService = Depends(),
    ):
        self.llm_service = llm_service
        self.rag_service = rag_service
        self.conversation_service = conversation_service

    async def voice_chat(self, user_id: int, text: str, agent: Agent) -> str:
        """
        Handles the voice chat interaction.
        1. Gets conversation history.
        2. (Optional) Uses RAG to get context.
        3. Generates a response from the LLM.
        4. Stores the new interaction in the conversation history.
        """
        try:
            # 1. Get conversation history
            history = await self.conversation_service.get_conversation_history(user_id, agent.id)
            
            # 2. RAG (if enabled for the agent)
            rag_context = ""
            if agent.rag_enabled:
                rag_context = await self.rag_service.search(
                    user_id=user_id,
                    query=text,
                    collection_name=agent.id # Assuming one collection per agent
                )

            # 3. Generate response
            llm_response = await self.llm_service.generate_response(
                prompt=text,
                system_prompt=agent.prompt,
                history=history,
                context=rag_context,
                model=agent.llm_model
            )
            
            # 4. Store conversation
            await self.conversation_service.store_conversation(
                user_id=user_id,
                agent_id=agent.id,
                user_message=text,
                agent_message=llm_response
            )
            
            return llm_response
            
        except Exception as e:
            logger.error(f"Error in voice_chat service: {e}")
            # In case of an error, return a generic response
            return "I'm sorry, I encountered an error and can't respond right now."

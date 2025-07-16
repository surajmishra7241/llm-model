import logging
from fastapi import Depends
from app.services.llm_service import OllamaService
from app.services.rag_service import RAGService
from app.services.conversation_service import ConversationService
from app.services.agent_service import AgentService
from app.models.agent_model import Agent
from app.config import settings

logger = logging.getLogger(__name__)

class VoiceAgentService:
    def __init__(
        self,
        llm_service: OllamaService = Depends(),
        rag_service: RAGService = Depends(),
        conversation_service: ConversationService = Depends(),
        agent_service: AgentService = Depends()
    ):
        self.llm_service = llm_service
        self.rag_service = rag_service
        self.conversation_service = conversation_service
        self.agent_service = agent_service

    async def voice_chat(
        self, 
        user_id: str, 
        text: str, 
        agent_id: str
    ) -> str:
        """
        Full voice chat workflow:
        1. Retrieve agent configuration
        2. Get conversation history
        3. Retrieve relevant context from RAG
        4. Generate LLM response
        5. Save interaction to memory
        """
        try:
            # 1. Get agent configuration
            agent = await self.agent_service.get_agent(agent_id, user_id)
            if not agent:
                logger.error(f"Agent {agent_id} not found for user {user_id}")
                return "I'm sorry, I can't find my configuration."
            
            # 2. Get conversation history
            history = await self.conversation_service.get_conversation_history(
                user_id, 
                agent_id,
                max_messages=settings.CONVERSATION_HISTORY_LIMIT
            )
            
            # 3. Retrieve RAG context if enabled
            rag_context = ""
            if agent.rag_enabled:
                rag_results = await self.rag_service.query(
                    user_id=user_id,
                    query=text,
                    agent_id=agent_id
                )
                rag_context = "\n".join(rag_results.get("documents", []))
                logger.debug(f"RAG context: {rag_context[:200]}...")
            
            # 4. Generate LLM response
            llm_response = await self.llm_service.chat(
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt(agent, rag_context)
                    },
                    *history,
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                model=agent.model,
                options={"temperature": 0.7}
            )
            
            response_text = llm_response.get("message", {}).get("content", "")
            
            # 5. Save interaction
            await self.conversation_service.save_interaction(
                user_id=user_id,
                agent_id=agent_id,
                user_message=text,
                agent_response=response_text,
                metadata={
                    "rag_context": rag_context,
                    "source": "voice_chat"
                }
            )
            
            return response_text
            
        except Exception as e:
            logger.error(f"Voice chat error: {str(e)}", exc_info=True)
            return "I'm having trouble responding right now. Please try again later."
    
    def _build_system_prompt(self, agent: Agent, rag_context: str) -> str:
        """Construct dynamic system prompt"""
        prompt_lines = [
            f"You are {agent.name}, {agent.description}",
            agent.system_prompt,
            "Current conversation context:"
        ]
        
        # Add RAG context if available
        if rag_context:
            prompt_lines.append("\nRelevant knowledge:")
            prompt_lines.append(rag_context[:2000])  # Limit context size
        
        # Voice-specific instructions
        prompt_lines.extend([
            "\nVoice Interaction Guidelines:",
            "- Respond conversationally as if speaking",
            "- Keep responses concise (1-2 sentences)",
            "- Use natural pauses and contractions",
            "- Adapt to conversational flow"
        ])
        
        return "\n".join(prompt_lines)
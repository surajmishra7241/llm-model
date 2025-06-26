# ./app/services/agent_interaction_service.py
import logging
from typing import Dict, Optional, List, AsyncIterator, Any  # Added Any here
from fastapi import UploadFile
from app.services.llm_service import OllamaService
from app.services.voice_service import VoiceService
from app.services.rag_service import RAGService
from app.models.agent_model import AgentPersonality
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

logger = logging.getLogger(__name__)

class EmotionalAnalyzer:
    def __init__(self, ollama_service: OllamaService):
        self.ollama = ollama_service
        
    async def analyze_emotion(self, text: str) -> Dict[str, float]:
        """Analyze emotional tone from text"""
        prompt = """Analyze the emotional tone of this text. Respond ONLY with a JSON object containing 
        emotion scores between 0-1 for: happiness, sadness, anger, fear, surprise, neutral.

        Text: {text}""".format(text=text)
        
        try:
            response = await self.ollama.generate(
                prompt=prompt,
                model="deepseek-r1:1.5b",
                options={"temperature": 0.2}
            )
            
            # Parse the JSON response
            import json
            return json.loads(response["response"])
        except Exception as e:
            logger.error(f"Emotion analysis failed: {str(e)}")
            return {
                "happiness": 0.5,
                "sadness": 0.0,
                "anger": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "neutral": 0.5
            }

class AgentInteractionService:
    def __init__(self):
        self.ollama = OllamaService()
        self.voice = VoiceService()
        self.rag = RAGService()
        self.emotion_analyzer = EmotionalAnalyzer(self.ollama)
        self.conversation_memory = {}
        
    async def initialize(self):
        """Initialize all required services"""
        await self.rag.initialize()
        
    async def process_input(
        self,
        agent_id: str,
        user_id: str,
        input_text: Optional[str] = None,
        audio_file: Optional[UploadFile] = None,
        db: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Process user input (text or voice) and generate response"""
        # Convert audio to text if provided
        if audio_file:
            input_text = await self.voice.speech_to_text(audio_file)
            
        if not input_text:
            raise ValueError("No input text provided")
            
        # Get or create conversation memory
        if user_id not in self.conversation_memory:
            self.conversation_memory[user_id] = {
                "history": [],
                "last_updated": datetime.now(),
                "emotional_state": {}
            }
            
        # Analyze emotion
        emotion_scores = await self.emotion_analyzer.analyze_emotion(input_text)
        self.conversation_memory[user_id]["emotional_state"] = emotion_scores
        
        # Retrieve relevant context
        rag_results = await self.rag.query(
            db=db,
            user_id=user_id,
            query=input_text
        )
        
        # Generate response considering emotion and context
        response = await self._generate_response(
            agent_id=agent_id,
            user_id=user_id,
            input_text=input_text,
            emotion_scores=emotion_scores,
            context=rag_results.get("documents", [])
        )
        
        # Update conversation history
        self.conversation_memory[user_id]["history"].append({
            "input": input_text,
            "response": response,
            "timestamp": datetime.now()
        })
        
        return {
            "text_response": response,
            "emotional_state": emotion_scores,
            "context_used": rag_results.get("sources", [])
        }
        
    async def _generate_response(
        self,
        agent_id: str,
        user_id: str,
        input_text: str,
        emotion_scores: Dict[str, float],
        context: List[str]
    ) -> str:
        """Generate response considering emotional state and context"""
        # Get dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        # Build system prompt based on emotion and context
        system_prompt = self._build_system_prompt(
            agent_id=agent_id,
            dominant_emotion=dominant_emotion,
            context=context
        )
        
        # Generate response
        response = await self.ollama.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ]
        )
        
        return response.get("message", {}).get("content", "")
        
    def _build_system_prompt(
        self,
        agent_id: str,
        dominant_emotion: str,
        context: List[str]
    ) -> str:
        """Build dynamic system prompt based on context and emotion"""
        # TODO: Fetch agent personality from database
        personality = AgentPersonality()  # Default personality
        
        # Base prompt
        prompt_lines = [
            f"You are {agent_id}, a highly intelligent AI assistant.",
            f"Your personality traits: {', '.join([t.value for t in personality.traits])}.",
            f"Current user emotional state: {dominant_emotion}. Adjust your tone accordingly.",
            "",
            "Context from knowledge base:",
            "\n".join(context) if context else "No relevant context found",
            "",
            "Guidelines:",
            f"- Be {personality.base_tone}",
            "- Acknowledge user's emotional state if strong",
            "- Use context when relevant but don't force it",
            "- Keep responses concise but thorough when needed",
            "- Maintain natural conversation flow"
        ]
        
        # Emotion-specific adjustments
        if dominant_emotion == "sadness":
            prompt_lines.append("- Show extra empathy and support")
        elif dominant_emotion == "anger":
            prompt_lines.append("- Remain calm and solution-focused")
        elif dominant_emotion == "happiness":
            prompt_lines.append("- Match the positive energy but stay professional")
            
        return "\n".join(prompt_lines)
        
    async def text_to_speech(
        self,
        text: str,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> bytes:
        """Convert text to speech with emotional inflection"""
        # TODO: Adjust TTS parameters based on emotional state
        return await self.voice.text_to_speech(text)
        
    async def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        return self.conversation_memory.get(user_id, {}).get("history", [])
        
    async def clear_memory(self, user_id: str):
        """Clear conversation memory for a user"""
        if user_id in self.conversation_memory:
            del self.conversation_memory[user_id]
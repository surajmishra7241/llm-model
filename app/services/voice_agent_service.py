# app/services/voice_websocket_service.py
import asyncio
import logging
import base64
import json
import time
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
from app.services.voice_service import VoiceService
from app.services.llm_service import OllamaService
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

class VoiceWebSocketService:
    def __init__(self):
        self.voice_service = VoiceService()
        self.llm_service = OllamaService()
        self.rag_service = RAGService()
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def handle_voice_websocket(self, websocket: WebSocket, connection_id: str):
        """Handle WebSocket voice interaction with optimized processing"""
        try:
            await websocket.accept()
            self.active_connections[connection_id] = websocket
            logger.info(f"Voice WebSocket connected: {connection_id}")
            
            while True:
                # Receive message with timeout
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_text(), 
                        timeout=300.0  # 5 minutes timeout
                    )
                    data = json.loads(message)
                    
                    await self.process_voice_message(websocket, connection_id, data)
                    
                except asyncio.TimeoutError:
                    logger.warning(f"WebSocket timeout for {connection_id}")
                    break
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await self.send_error(websocket, "Invalid JSON format")
                    
        except Exception as e:
            logger.error(f"Voice WebSocket error for {connection_id}: {str(e)}")
        finally:
            self.active_connections.pop(connection_id, None)
            logger.info(f"Voice WebSocket disconnected: {connection_id}")

    async def process_voice_message(self, websocket: WebSocket, connection_id: str, data: Dict[str, Any]):
        """Process voice message with optimized pipeline"""
        message_type = data.get("type")
        
        if message_type == "voice_input":
            await self.process_voice_input(websocket, connection_id, data)
        elif message_type == "ping":
            await websocket.send_text(json.dumps({"type": "pong"}))
        else:
            await self.send_error(websocket, f"Unknown message type: {message_type}")

    async def process_voice_input(self, websocket: WebSocket, connection_id: str, data: Dict[str, Any]):
        """Optimized voice input processing with parallel operations"""
        start_time = time.time()
        
        try:
            # Extract data
            audio_data = data.get("audio_data", {})
            user_id = data.get("user_id")
            agent_id = data.get("agent_id")
            voice_config = data.get("voice_config", {})
            personality = data.get("personality", {})
            system_prompt = data.get("system_prompt", "")
            optimization = data.get("optimization", {})
            
            # Validate required fields
            if not audio_data.get("content"):
                await self.send_error(websocket, "No audio data provided")
                return
                
            # Send processing status
            await websocket.send_text(json.dumps({
                "type": "processing",
                "step": "transcribing"
            }))
            
            # Step 1: Speech to Text (optimized)
            stt_start = time.time()
            audio_content = base64.b64decode(audio_data["content"])
            
            stt_result = await self.voice_service.speech_to_text_optimized(
                audio_data=audio_content,
                format=audio_data.get("format", "webm"),
                user_id=user_id,
                optimization_settings={
                    "fast_mode": optimization.get("reduce_latency", True),
                    "language": "en-US",
                    "model_size": "base"  # Use smaller model for speed
                }
            )
            
            transcription = stt_result.get("text", "").strip()
            confidence = stt_result.get("confidence", 0.0)
            stt_time = time.time() - stt_start
            
            # Send transcription immediately
            await websocket.send_text(json.dumps({
                "type": "transcription",
                "text": transcription,
                "confidence": confidence,
                "processing_time": round(stt_time, 3)
            }))
            
            if not transcription:
                await self.send_error(websocket, "Could not transcribe audio")
                return
            
            # Send processing status
            await websocket.send_text(json.dumps({
                "type": "processing",
                "step": "generating_response"
            }))
            
            # Step 2: Parallel RAG and LLM processing
            llm_start = time.time()
            
            # Start RAG search and LLM processing in parallel
            rag_task = self.get_rag_context(transcription, user_id, agent_id)
            llm_task = self.generate_base_response(transcription, system_prompt, personality)
            
            # Wait for both tasks with timeout
            try:
                rag_context, base_response = await asyncio.wait_for(
                    asyncio.gather(rag_task, llm_task, return_exceptions=True),
                    timeout=10.0  # 10 second timeout for LLM+RAG
                )
            except asyncio.TimeoutError:
                logger.warning(f"LLM/RAG processing timeout for {connection_id}")
                rag_context = ""
                base_response = "I apologize, but I'm experiencing some delays. Please try again."
            
            # Handle exceptions from parallel tasks
            if isinstance(rag_context, Exception):
                logger.error(f"RAG processing error: {str(rag_context)}")
                rag_context = ""
                
            if isinstance(base_response, Exception):
                logger.error(f"LLM processing error: {str(base_response)}")
                base_response = "I apologize, but I'm having trouble generating a response right now."
            
            # Step 3: Enhance response with RAG context if available
            if rag_context:
                enhanced_response = await self.enhance_response_with_rag(
                    base_response, rag_context, transcription
                )
            else:
                enhanced_response = base_response
                
            llm_time = time.time() - llm_start
            
            # Step 4: Text to Speech (optimized)
            await websocket.send_text(json.dumps({
                "type": "processing",
                "step": "generating_speech"
            }))
            
            tts_start = time.time()
            tts_result = await self.voice_service.text_to_speech_optimized(
                text=enhanced_response,
                user_id=user_id,
                voice_config=voice_config,
                optimization_settings={
                    "quality": optimization.get("audio_quality", "balanced"),
                    "speed": optimization.get("reduce_latency", True)
                }
            )
            tts_time = time.time() - tts_start
            
            # Step 5: Send complete response
            total_time = time.time() - start_time
            
            await websocket.send_text(json.dumps({
                "type": "agent_response",
                "text": enhanced_response,
                "audio": tts_result.get("audio_base64"),
                "processing_time": {
                    "total": round(total_time, 3),
                    "stt": round(stt_time, 3),
                    "llm": round(llm_time, 3),
                    "tts": round(tts_time, 3)
                },
                "transcription": transcription,
                "confidence": confidence,
                "sources": rag_context.get("sources", []) if isinstance(rag_context, dict) else []
            }))
            
            logger.info(f"Voice interaction completed in {total_time:.3f}s for {connection_id}")
            
        except Exception as e:
            logger.error(f"Voice processing error for {connection_id}: {str(e)}")
            await self.send_error(websocket, "Voice processing failed")

    async def get_rag_context(self, query: str, user_id: str, agent_id: str) -> Dict[str, Any]:
        """Get RAG context with timeout"""
        try:
            rag_results = await asyncio.wait_for(
                self.rag_service.query_optimized(
                    query=query,
                    user_id=user_id,
                    agent_id=agent_id,
                    max_results=3,
                    optimization={"fast_mode": True}
                ),
                timeout=5.0
            )
            return rag_results
        except asyncio.TimeoutError:
            logger.warning("RAG query timeout")
            return {}
        except Exception as e:
            logger.error(f"RAG query error: {str(e)}")
            return {}

    async def generate_base_response(self, query: str, system_prompt: str, personality: Dict) -> str:
        """Generate base LLM response"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            response = await self.llm_service.chat_optimized(
                messages=messages,
                model="deepseek-r1:1.5b",
                options={
                    "temperature": personality.get("base_tone_temperature", 0.7),
                    "max_tokens": 150,  # Shorter responses for voice
                    "stream": False
                }
            )
            
            return response.get("message", {}).get("content", "I apologize, I couldn't generate a response.")
            
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            return "I'm experiencing some technical difficulties. Please try again."

    async def enhance_response_with_rag(self, base_response: str, rag_context: Dict, query: str) -> str:
        """Enhance response with RAG context"""
        try:
            if not rag_context.get("documents"):
                return base_response
                
            context_text = "\n".join(rag_context["documents"][:2])  # Limit context
            
            enhancement_prompt = f"""
            Original response: {base_response}
            
            Additional context: {context_text}
            
            User question: {query}
            
            Enhance the original response with relevant information from the context, but keep it concise for voice conversation:
            """
            
            enhanced = await self.llm_service.chat_optimized(
                messages=[{"role": "user", "content": enhancement_prompt}],
                model="deepseek-r1:1.5b",
                options={"temperature": 0.3, "max_tokens": 200}
            )
            
            return enhanced.get("message", {}).get("content", base_response)
            
        except Exception as e:
            logger.error(f"Response enhancement error: {str(e)}")
            return base_response

    async def send_error(self, websocket: WebSocket, message: str):
        """Send error message via WebSocket"""
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": message
            }))
        except Exception as e:
            logger.error(f"Failed to send error message: {str(e)}")

# Updated WebSocket endpoint
@router.websocket("/ws")
async def voice_websocket_endpoint(websocket: WebSocket):
    connection_id = f"voice_{int(time.time())}_{hash(websocket) % 10000}"
    voice_service = VoiceWebSocketService()
    await voice_service.handle_voice_websocket(websocket, connection_id)

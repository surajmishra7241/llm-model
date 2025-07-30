# app/routers/voice_websocket_streaming.py
import asyncio
import json
import logging
import time
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.streaming_stt_service import StreamingSTTService
from app.services.streaming_tts_service import StreamingTTSService
from app.services.llm_service import OllamaService

router = APIRouter()
logger = logging.getLogger(__name__)

class StreamingVoiceHandler:
    def __init__(self):
        self.stt_service = StreamingSTTService()
        self.tts_service = StreamingTTSService()
        self.llm_service = OllamaService()
        self.active_sessions = {}

    async def initialize(self):
        """Initialize all services"""
        await self.stt_service.initialize()
        await self.tts_service.initialize()

    async def handle_websocket_connection(self, websocket: WebSocket, session_id: str):
        """Handle streaming voice WebSocket connection"""
        try:
            await websocket.accept()
            logger.info(f"Voice streaming session started: {session_id}")
            
            # Store session
            self.active_sessions[session_id] = {
                'websocket': websocket,
                'start_time': time.time(),
                'message_count': 0
            }

            await websocket.send_text(json.dumps({
                'type': 'session_started',
                'session_id': session_id,
                'timestamp': time.time()
            }))

            # Message loop
            async for message in websocket.iter_text():
                try:
                    data = json.loads(message)
                    await self._handle_message(websocket, session_id, data)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Message handling error: {e}")
                    await self._send_error(websocket, str(e))

        except WebSocketDisconnect:
            logger.info(f"Voice session disconnected: {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await self._cleanup_session(session_id)

    async def _handle_message(self, websocket: WebSocket, session_id: str, data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        message_type = data.get('type')

        if message_type == 'audio_chunk':
            await self._handle_audio_chunk(websocket, session_id, data)
        elif message_type == 'text_input':
            await self._handle_text_input(websocket, session_id, data)
        elif message_type == 'ping':
            await websocket.send_text(json.dumps({
                'type': 'pong',
                'timestamp': time.time()
            }))
        else:
            await self._send_error(websocket, f"Unknown message type: {message_type}")

    async def _handle_audio_chunk(self, websocket: WebSocket, session_id: str, data: Dict[str, Any]):
        """Handle streaming audio chunk"""
        try:
            audio_data = data.get('audio_data')
            if not audio_data:
                await self._send_error(websocket, "No audio data provided")
                return

            # Process STT
            stt_result = await self.stt_service.process_audio_chunk(audio_data, session_id)
            
            # Send transcription result
            await websocket.send_text(json.dumps({
                'type': 'transcription',
                'transcription': stt_result.get('text', ''),
                'is_final': stt_result.get('type') == 'final',
                'confidence': stt_result.get('confidence', 0),
                'session_id': session_id,
                'timestamp': time.time()
            }))

            # If final transcription and has text, generate response
            if stt_result.get('type') == 'final' and stt_result.get('text', '').strip():
                await self._generate_agent_response(websocket, session_id, stt_result.get('text'))

        except Exception as e:
            logger.error(f"Audio chunk handling error: {e}")
            await self._send_error(websocket, str(e))

    async def _handle_text_input(self, websocket: WebSocket, session_id: str, data: Dict[str, Any]):
        """Handle direct text input"""
        try:
            text = data.get('text', '').strip()
            if not text:
                await self._send_error(websocket, "No text provided")
                return

            await self._generate_agent_response(websocket, session_id, text)

        except Exception as e:
            logger.error(f"Text input handling error: {e}")
            await self._send_error(websocket, str(e))

    async def _generate_agent_response(self, websocket: WebSocket, session_id: str, user_text: str):
        """Generate and stream agent response"""
        try:
            # Send processing started
            await websocket.send_text(json.dumps({
                'type': 'processing_started',
                'session_id': session_id,
                'timestamp': time.time()
            }))

            # Generate LLM response
            messages = [
                {"role": "system", "content": "You are a helpful voice assistant. Keep responses conversational and under 100 words."},
                {"role": "user", "content": user_text}
            ]

            llm_response = await self.llm_service.chat(
                messages=messages,
                model="deepseek-r1:1.5b",
                options={"temperature": 0.7, "max_tokens": 150}
            )

            response_text = llm_response.get("message", {}).get("content", "I'm sorry, I couldn't generate a response.")

            # Send text response
            await websocket.send_text(json.dumps({
                'type': 'agent_response',
                'text': response_text,
                'session_id': session_id,
                'timestamp': time.time()
            }))

            # Generate and stream TTS
            await websocket.send_text(json.dumps({
                'type': 'tts_started',
                'session_id': session_id
            }))

            async for audio_chunk in self.tts_service.text_to_speech_stream(
                response_text, 
                session_id=session_id
            ):
                await websocket.send_text(json.dumps(audio_chunk))

            await websocket.send_text(json.dumps({
                'type': 'tts_completed',
                'session_id': session_id,
                'timestamp': time.time()
            }))

        except Exception as e:
            logger.error(f"Agent response generation error: {e}")
            await self._send_error(websocket, str(e))

    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message"""
        try:
            await websocket.send_text(json.dumps({
                'type': 'error',
                'message': error_message,
                'timestamp': time.time()
            }))
        except Exception as e:
            logger.error(f"Failed to send error: {e}")

    async def _cleanup_session(self, session_id: str):
        """Cleanup session resources"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        logger.info(f"Session cleaned up: {session_id}")

# Global handler instance
streaming_handler = StreamingVoiceHandler()

@router.on_event("startup")
async def startup_event():
    await streaming_handler.initialize()

@router.websocket("/voice/stream")
async def voice_streaming_endpoint(websocket: WebSocket):
    """Streaming voice WebSocket endpoint"""
    import uuid
    session_id = f"voice_stream_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    await streaming_handler.handle_websocket_connection(websocket, session_id)

# app/routers/voice_websocket.py - Fixed imports and error handling
import asyncio
import json
import time
import logging
import uuid
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.ultra_fast_voice_service import UltraFastVoiceService
from app.services.connection_manager import ConnectionManager
from app.services.voice_session_manager import VoiceSessionManager
from app.utils.performance_monitor import PerformanceMonitor
from app.utils.advanced_vad import AdvancedVAD

router = APIRouter()
logger = logging.getLogger(__name__)

# Global managers
connection_manager = ConnectionManager()
session_manager = VoiceSessionManager()
performance_monitor = PerformanceMonitor()

class UltraFastWebSocketHandler:
    def __init__(self):
        self.voice_service = UltraFastVoiceService()
        self.vad_configs = {
            "ultra_fast": {
                "silenceThreshold": 0.01,
                "speechThreshold": 0.025,
                "minSilenceDuration": 600,
                "minSpeechDuration": 150,
                "maxRecordingTime": 12000,
                "vadSensitivity": 0.8,
                "endpointDetection": True
            }
        }

    async def handle_connection(self, websocket: WebSocket, connection_id: str):
        """Handle new WebSocket connection with ultra-fast setup"""
        try:
            await websocket.accept()
            logger.info(f"Ultra-fast voice WebSocket connected: {connection_id}")
            
            # Initialize connection
            await connection_manager.add_connection(connection_id, websocket)
            
            # Send immediate connection confirmation
            await self._send_message(websocket, {
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": time.time(),
                "performance_mode": "ultra_fast"
            })
            
            # Message handling loop
            async for message in websocket.iter_text():
                try:
                    data = json.loads(message)
                    await self._handle_message(connection_id, websocket, data)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    logger.error(f"Message handling error: {str(e)}")
                    await self._send_error(websocket, f"Message processing failed: {str(e)}")
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
        finally:
            await self._cleanup_connection(connection_id)

    async def _handle_message(self, connection_id: str, websocket: WebSocket, data: Dict[str, Any]):
        """Handle incoming WebSocket messages"""
        message_type = data.get("type")
        timestamp = time.time()
        
        try:
            if message_type == "authenticate":
                await self._handle_authentication(connection_id, websocket, data)
            elif message_type == "voice_input":
                await self._handle_voice_input(connection_id, websocket, data, timestamp)
            elif message_type == "text_input":
                await self._handle_text_input(connection_id, websocket, data, timestamp)
            elif message_type == "ping":
                await self._handle_ping(websocket, data.get("timestamp"))
            else:
                await self._send_error(websocket, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Message handling error for {message_type}: {str(e)}")
            await self._send_error(websocket, f"Failed to process {message_type}")

    async def _handle_authentication(self, connection_id: str, websocket: WebSocket, data: Dict[str, Any]):
        """Handle authentication with session setup"""
        try:
            user_id = data.get("user_id")
            agent_id = data.get("agent_id")
            
            if not user_id or not agent_id:
                await self._send_error(websocket, "user_id and agent_id are required")
                return
            
            # Create voice session (with minimal agent data for now)
            mock_agent = type('MockAgent', (), {
                '_id': agent_id,
                'name': 'Test Agent',
                'systemPrompt': 'You are a helpful assistant.'
            })()
            
            session = await session_manager.create_session(
                connection_id, user_id, agent_id, websocket, mock_agent
            )
            
            # Initialize voice service for this session
            await self.voice_service.initialize_session(session)
            
            await self._send_message(websocket, {
                "type": "authenticated",
                "session_id": connection_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "performance_targets": {
                    "total_response_time": "< 1200ms",
                    "stt_time": "< 300ms",
                    "llm_time": "< 500ms", 
                    "tts_time": "< 400ms"
                }
            })
            
            logger.info(f"Session authenticated: {connection_id} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            await self._send_error(websocket, f"Authentication failed: {str(e)}")

    async def _handle_voice_input(self, connection_id: str, websocket: WebSocket, data: Dict[str, Any], timestamp: float):
        """Handle voice input with ultra-fast processing"""
        try:
            session = await session_manager.get_session(connection_id)
            if not session:
                await self._send_error(websocket, "Session not authenticated")
                return
            
            audio_data = data.get("audio_data")
            if not audio_data:
                await self._send_error(websocket, "No audio data provided")
                return
            
            # Send processing acknowledgment
            await self._send_message(websocket, {
                "type": "processing_started",
                "timestamp": timestamp,
                "estimated_completion": timestamp + 1.2
            })
            
            # Process voice with ultra-fast service
            result = await self.voice_service.process_voice_ultra_fast(
                session=session,
                audio_data=audio_data,
                audio_format=data.get("format", "webm"),
                voice_config=data.get("voice_config", {}),
                processing_options=data.get("processing_options", {})
            )
            
            # Send response
            await self._send_message(websocket, {
                "type": "voice_response",
                "transcription": result["transcription"],
                "response_text": result["response"]["text"],
                "audio_data": result["response"]["audio"],
                "audio_format": result["response"].get("format", "wav"),
                "confidence": result["confidence"],
                "processing_time": result["processing_time"],
                "performance_metrics": result.get("performance_metrics", {}),
                "sources": result.get("sources", {}),
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"Voice input error: {str(e)}")
            await self._send_error(websocket, f"Voice processing failed: {str(e)}")

    async def _handle_text_input(self, connection_id: str, websocket: WebSocket, data: Dict[str, Any], timestamp: float):
        """Handle text input"""
        try:
            session = await session_manager.get_session(connection_id)
            if not session:
                await self._send_error(websocket, "Session not authenticated")
                return
            
            text = data.get("text", "").strip()
            if not text:
                await self._send_error(websocket, "No text provided")
                return
            
            result = await self.voice_service.process_text_ultra_fast(
                session=session,
                text=text,
                processing_options=data.get("processing_options", {})
            )
            
            await self._send_message(websocket, {
                "type": "text_response",
                "input": text,
                "response": result["response"],
                "processing_time": result["processing_time"],
                "sources": result.get("sources", {}),
                "timestamp": time.time()
            })
            
        except Exception as e:
            logger.error(f"Text input error: {str(e)}")
            await self._send_error(websocket, f"Text processing failed: {str(e)}")

    async def _handle_ping(self, websocket: WebSocket, client_timestamp: float = None):
        """Handle ping with latency measurement"""
        server_timestamp = time.time()
        latency = (server_timestamp - client_timestamp) * 1000 if client_timestamp else 0
        
        await self._send_message(websocket, {
            "type": "pong",
            "client_timestamp": client_timestamp,
            "server_timestamp": server_timestamp,
            "latency_ms": round(latency, 2) if client_timestamp else None
        })

    async def _send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message with error handling"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send message: {str(e)}")

    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error message"""
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": error_message,
                "timestamp": time.time()
            }))
        except Exception as e:
            logger.error(f"Failed to send error: {str(e)}")

    async def _cleanup_connection(self, connection_id: str):
        """Cleanup connection resources"""
        try:
            await session_manager.remove_session(connection_id)
            await connection_manager.remove_connection(connection_id)
            logger.info(f"Connection cleaned up: {connection_id}")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")

# Global handler instance
websocket_handler = UltraFastWebSocketHandler()

@router.websocket("/voice/ws")
async def voice_websocket_endpoint(websocket: WebSocket):
    """Ultra-fast voice WebSocket endpoint"""
    connection_id = f"voice_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    await websocket_handler.handle_connection(websocket, connection_id)

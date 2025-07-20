# app/routers/voice_websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List
import asyncio
import json
import logging
import base64
import io
import wave
from app.services.voice_service import VoiceService
from app.services.llm_service import OllamaService
from app.dependencies import get_current_user, get_db
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

router = APIRouter(prefix="/voice", tags=["Voice WebSocket"])
logger = logging.getLogger(__name__)

class VoiceConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, List[str]] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = []
        self.user_connections[user_id].append(connection_id)
        
        logger.info(f"Voice WebSocket connected: {connection_id} for user {user_id}")
    
    def disconnect(self, connection_id: str, user_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_connections:
            if connection_id in self.user_connections[user_id]:
                self.user_connections[user_id].remove(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"Voice WebSocket disconnected: {connection_id}")
    
    async def send_message(self, connection_id: str, message: dict):
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                # Remove dead connection
                self.disconnect(connection_id, "unknown")

manager = VoiceConnectionManager()

@router.websocket("/ws")
async def voice_websocket_endpoint(websocket: WebSocket):
    connection_id = str(uuid.uuid4())
    user_id = None
    
    try:
        await websocket.accept()
        logger.info(f"Voice WebSocket connection established: {connection_id}")
        
        # Wait for authentication message
        auth_data = await websocket.receive_text()
        auth_message = json.loads(auth_data)
        
        if auth_message.get("type") != "authenticate":
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Authentication required"
            }))
            await websocket.close()
            return
        
        # Verify authentication token
        token = auth_message.get("token")
        agent_id = auth_message.get("agent_id")
        
        if not token or not agent_id:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Token and agent_id required"
            }))
            await websocket.close()
            return
        
        # TODO: Add proper token verification here
        user_id = "extracted_from_token"  # Replace with actual token verification
        
        # Store connection
        manager.active_connections[connection_id] = websocket
        if user_id not in manager.user_connections:
            manager.user_connections[user_id] = []
        manager.user_connections[user_id].append(connection_id)
        
        # Send authentication success
        await websocket.send_text(json.dumps({
            "type": "auth_success",
            "connection_id": connection_id
        }))
        
        # Initialize services
        voice_service = VoiceService()
        llm_service = OllamaService()
        
        # Main message loop
        while True:
            try:
                # Receive message
                message_data = await websocket.receive_text()
                message = json.loads(message_data)
                
                await handle_voice_message(
                    websocket, 
                    connection_id, 
                    user_id, 
                    agent_id, 
                    message,
                    voice_service,
                    llm_service
                )
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Processing error"
                }))
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if user_id:
            manager.disconnect(connection_id, user_id)

async def handle_voice_message(
    websocket: WebSocket,
    connection_id: str,
    user_id: str,
    agent_id: str,
    message: dict,
    voice_service: VoiceService,
    llm_service: OllamaService
):
    """Handle different types of voice messages"""
    message_type = message.get("type")
    
    if message_type == "voice_input":
        await process_voice_input(
            websocket, connection_id, user_id, agent_id, message, 
            voice_service, llm_service
        )
    elif message_type == "ping":
        await websocket.send_text(json.dumps({"type": "pong"}))
    else:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        }))

async def process_voice_input(
    websocket: WebSocket,
    connection_id: str,
    user_id: str,
    agent_id: str,
    message: dict,
    voice_service: VoiceService,
    llm_service: OllamaService
):
    """Process voice input and generate response"""
    try:
        # Extract audio data
        audio_base64 = message.get("audio")
        audio_format = message.get("format", "webm")
        
        if not audio_base64:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "No audio data provided"
            }))
            return
        
        # Decode audio data
        audio_data = base64.b64decode(audio_base64)
        
        # Step 1: Speech to Text
        await websocket.send_text(json.dumps({
            "type": "processing",
            "step": "transcribing"
        }))
        
        stt_result = await voice_service.speech_to_text(
            audio_data=audio_data,
            format=audio_format,
            user_id=user_id
        )
        
        transcription = stt_result.get("text", "")
        confidence = stt_result.get("confidence", 0.0)
        
        # Send transcription result
        await websocket.send_text(json.dumps({
            "type": "transcription",
            "text": transcription,
            "confidence": confidence
        }))
        
        if not transcription:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Could not transcribe audio"
            }))
            return
        
        # Step 2: Generate Agent Response
        await websocket.send_text(json.dumps({
            "type": "processing",
            "step": "generating_response"
        }))
        
        # Get agent response through LLM
        agent_response = await llm_service.chat(
            messages=[
                {"role": "user", "content": transcription}
            ],
            model="deepseek-r1:1.5b",  # Or get from agent config
            agent_id=agent_id,
            user_id=user_id
        )
        
        response_text = agent_response.get("message", {}).get("content", "")
        
        # Step 3: Text to Speech
        await websocket.send_text(json.dumps({
            "type": "processing",
            "step": "generating_speech"
        }))
        
        tts_result = await voice_service.text_to_speech(
            text=response_text,
            user_id=user_id,
            agent_id=agent_id
        )
        
        # Send complete response
        await websocket.send_text(json.dumps({
            "type": "agent_response",
            "text": response_text,
            "audio": tts_result.get("audio_base64"),
            "processing_time": tts_result.get("processing_time"),
            "transcription": transcription,
            "confidence": confidence
        }))
        
    except Exception as e:
        logger.error(f"Error processing voice input: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Failed to process voice input"
        }))

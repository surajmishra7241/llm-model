# app/routers/voice_processing.py - Fixed for Node.js integration
import asyncio
import time
import base64
import json
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.services.voice_service import VoiceService
from app.services.llm_service import OllamaService
from app.dependencies import get_current_user
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
voice_service = VoiceService()
llm_service = OllamaService()

@router.post("/voice/process-streaming")
async def process_streaming_audio(
    request: dict,
    user: dict = Depends(get_current_user)
):
    """Process streaming audio from Node.js WebSocket - FIXED"""
    try:
        audio_data = request.get("audio_data")
        format_type = request.get("format", "webm")
        session_id = request.get("session_id")
        user_id = request.get("user_id")
        agent_id = request.get("agent_id")
        is_final = request.get("is_final", False)
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        logger.info(f"Processing streaming audio for session {session_id}")
        
        # Ensure voice service is initialized
        await voice_service._ensure_models_loaded()
        
        # Convert base64 to bytes
        try:
            audio_bytes = base64.b64decode(audio_data)
        except Exception as e:
            logger.error(f"Failed to decode audio data: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid audio data format")
        
        # Process with Whisper STT
        result = await voice_service.speech_to_text(
            audio_data=audio_bytes, 
            format=format_type, 
            user_id=user_id
        )
        
        return {
            "success": True,
            "transcription": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
            "is_final": is_final,
            "session_id": session_id,
            "processing_time": result.get("processing_time", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Streaming audio processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/voice/tts")
async def text_to_speech_endpoint(
    request: dict,
    user: dict = Depends(get_current_user)
):
    """Convert text to speech - FIXED"""
    try:
        text = request.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")
        
        logger.info(f"🔊 TTS Request: {text[:50]}...")
        
        # Initialize voice service
        await voice_service._ensure_models_loaded()
        
        # Generate TTS
        result = await voice_service.text_to_speech(
            text=text,
            user_id=request.get("user_id"),
            agent_id=request.get("agent_id"),
            voice_config=request.get("voice_config", {})
        )
        
        if not result.get("audio_base64"):
            raise HTTPException(status_code=500, detail="TTS generation failed")
        
        logger.info(f"✅ TTS Success: {len(result['audio_base64'])} chars")
        
        return {
            "success": True,
            "audio_base64": result["audio_base64"],
            "format": "wav",
            "sample_rate": 16000
        }
        
    except Exception as e:
        logger.error(f"❌ TTS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice/health")
async def voice_service_health():
    """Health check for voice service"""
    try:
        status_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "whisper": "available",
                "tts": "available", 
                "ollama": "checking..."
            }
        }
        
        # Quick Ollama health check
        try:
            ollama_status = await llm_service.check_ollama_status()
            status_info["services"]["ollama"] = "healthy" if ollama_status.get("status") == "healthy" else "degraded"
        except:
            status_info["services"]["ollama"] = "unavailable"
        
        return status_info
        
    except Exception as e:
        logger.error(f"Voice health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.models.voice_model import VoiceResponse, TextToSpeechRequest
from app.services.voice_service import VoiceService
from app.dependencies import get_current_user
from fastapi.responses import StreamingResponse
import io

router = APIRouter()

@router.post("/stt", response_model=VoiceResponse)
async def speech_to_text(
    audio_file: UploadFile = File(...),
    voice_service: VoiceService = Depends(),
    user: dict = Depends(get_current_user)
):
    """Convert speech to text"""
    try:
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        text = await voice_service.speech_to_text(audio_file)
        return VoiceResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tts")
async def text_to_speech(
    request: TextToSpeechRequest,
    voice_service: VoiceService = Depends(),
    user: dict = Depends(get_current_user)
):
    """Convert text to speech"""
    try:
        audio_data = await voice_service.text_to_speech(request.text)
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
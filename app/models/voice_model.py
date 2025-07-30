# app/models/voice_model.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class AudioData(BaseModel):
    content: str = Field(..., description="Base64 encoded audio data")
    format: str = Field(default="webm", description="Audio format")
    size: Optional[int] = Field(None, description="Audio data size in bytes")

class VoiceConfig(BaseModel):
    model_name: str = Field(default="default", description="Voice model to use")
    speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech rate")
    pitch: float = Field(default=0.0, ge=-20.0, le=20.0, description="Pitch adjustment")
    volume_gain: float = Field(default=0.0, ge=-10.0, le=10.0, description="Volume gain")
    voice: str = Field(default="alloy", description="Voice type")

class PersonalitySettings(BaseModel):
    traits: List[str] = Field(default_factory=list, description="Personality traits")
    base_tone: str = Field(default="friendly", description="Base tone of voice")
    emotional_awareness: Dict[str, Any] = Field(default_factory=dict)

class VoiceProcessingRequest(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio data")
    format: Optional[str] = Field(default="webm", description="Audio format")
    session_id: str = Field(..., description="WebSocket session ID")
    user_id: str = Field(..., description="User ID")
    agent_id: str = Field(..., description="Agent ID")
    is_final: Optional[bool] = Field(default=False, description="Is final audio chunk")
    voice_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    system_prompt: Optional[str] = Field(default="")
    personality: Optional[Dict[str, Any]] = Field(default_factory=dict)

class VoiceResponse(BaseModel):
    success: bool
    transcription: str
    confidence: float
    is_final: bool
    session_id: str
    processing_time: float

class TextToSpeechRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice_config: Optional[VoiceConfig] = Field(default_factory=VoiceConfig)
    user_id: Optional[str] = Field(None, description="User ID")
    agent_id: Optional[str] = Field(None, description="Agent ID")

class SpeechToTextRequest(BaseModel):
    audio_data: AudioData = Field(..., description="Audio data to transcribe")
    user_id: str = Field(..., description="User ID")
    language: Optional[str] = Field("en-US", description="Expected language")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict)

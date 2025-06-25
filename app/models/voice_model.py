from pydantic import BaseModel
from typing import Optional

class VoiceResponse(BaseModel):
    text: str

class TextToSpeechRequest(BaseModel):
    text: str
    voice_model: Optional[str] = None
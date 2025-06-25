import logging
from typing import Optional, Tuple
from fastapi import UploadFile
from app.config import settings
import io
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self):
        self.whisper_model = None
        self.tts_model = None
        self.load_models()

    def load_models(self):
        """Lazy load voice models"""
        try:
            if settings.ENABLE_WHISPER:
                import whisper
                self.whisper_model = whisper.load_model(settings.WHISPER_MODEL)
                
            if settings.ENABLE_TTS:
                if settings.TTS_ENGINE == "coqui":
                    from TTS.api import TTS
                    self.tts_model = TTS(model_name=settings.TTS_MODEL)
                elif settings.TTS_ENGINE == "pyttsx3":
                    import pyttsx3
                    self.tts_model = pyttsx3.init()
        except ImportError as e:
            logger.warning(f"Voice models not available: {str(e)}")

    async def speech_to_text(
        self,
        audio_file: UploadFile,
        language: Optional[str] = None
    ) -> str:
        """Convert speech to text using Whisper"""
        if not self.whisper_model:
            raise RuntimeError("Whisper is not enabled or failed to load")
            
        try:
            # Read audio file
            contents = await audio_file.read()
            audio_buffer = io.BytesIO(contents)
            
            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(audio_buffer)
            
            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000
                )
                waveform = resampler(waveform)
            
            # Convert to numpy array
            audio_np = waveform.numpy().squeeze()
            
            # Transcribe
            result = self.whisper_model.transcribe(
                audio_np,
                language=language,
                fp16=torch.cuda.is_available()
            )
            return result["text"]
        except Exception as e:
            logger.error(f"Error in speech-to-text: {str(e)}")
            raise

    async def text_to_speech(
        self,
        text: str,
        output_format: str = "wav"
    ) -> bytes:
        """Convert text to speech"""
        if not self.tts_model:
            raise RuntimeError("TTS is not enabled or failed to load")
            
        try:
            if settings.TTS_ENGINE == "coqui":
                # Coqui TTS
                output = io.BytesIO()
                self.tts_model.tts_to_file(
                    text=text,
                    file_path=output,
                    speaker=self.tts_model.speakers[0],
                    language=self.tts_model.languages[0]
                )
                return output.getvalue()
                
            elif settings.TTS_ENGINE == "pyttsx3":
                # pyttsx3 TTS
                output = io.BytesIO()
                self.tts_model.save_to_file(text, output)
                self.tts_model.runAndWait()
                return output.getvalue()
                
            else:
                raise ValueError("Unsupported TTS engine")
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
            raise
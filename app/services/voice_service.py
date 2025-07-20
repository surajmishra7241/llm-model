# app/services/voice_service.py
import asyncio
import logging
import base64
import io
import wave
import tempfile
import os
from typing import Optional, Dict, Any
from fastapi import UploadFile
import torch
import torchaudio
from app.config import settings

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self):
        self.whisper_model = None
        self.tts_model = None
        self.load_models()

    def load_models(self):
        """Initialize voice models"""
        try:
            if settings.ENABLE_WHISPER:
                import whisper
                self.whisper_model = whisper.load_model(settings.WHISPER_MODEL)
                logger.info(f"Loaded Whisper model: {settings.WHISPER_MODEL}")
                
            if settings.ENABLE_TTS:
                if settings.TTS_ENGINE == "coqui":
                    from TTS.api import TTS
                    self.tts_model = TTS(model_name=settings.TTS_MODEL)
                    logger.info(f"Loaded TTS model: {settings.TTS_MODEL}")
                    
        except ImportError as e:
            logger.warning(f"Voice models not available: {str(e)}")

    async def speech_to_text(
        self,
        audio_data: bytes,
        format: str = "webm",
        user_id: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert speech to text using Whisper"""
        if not self.whisper_model:
            raise RuntimeError("Whisper model not available")
            
        try:
            # Convert WebM to WAV if needed
            if format == "webm":
                audio_data = await self._convert_webm_to_wav(audio_data)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Load and preprocess audio
                waveform, sample_rate = torchaudio.load(temp_path)
                
                # Resample if needed (Whisper expects 16kHz)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000
                    )
                    waveform = resampler(waveform)
                
                # Convert to numpy for Whisper
                audio_np = waveform.numpy().squeeze()
                
                # Transcribe with Whisper
                result = self.whisper_model.transcribe(
                    audio_np,
                    language=language,
                    fp16=torch.cuda.is_available()
                )
                
                return {
                    "text": result["text"].strip(),
                    "confidence": self._calculate_confidence(result),
                    "language": result.get("language", "unknown"),
                    "segments": result.get("segments", [])
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Speech to text error: {str(e)}")
            raise RuntimeError(f"STT processing failed: {str(e)}")

    async def text_to_speech(
        self,
        text: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        voice_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Convert text to speech"""
        if not self.tts_model:
            raise RuntimeError("TTS model not available")
            
        try:
            # Get voice configuration
            config = voice_config or {}
            speaking_rate = config.get("speaking_rate", 1.0)
            pitch = config.get("pitch", 0.0)
            
            # Generate speech
            if settings.TTS_ENGINE == "coqui":
                # Generate audio with Coqui TTS
                audio_data = self.tts_model.tts(text)
                
                # Convert to base64 for transmission
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                return {
                    "audio_base64": audio_base64,
                    "format": "wav",
                    "sample_rate": 22050,
                    "processing_time": 0.0  # Add timing if needed
                }
            else:
                raise RuntimeError("Unsupported TTS engine")
                
        except Exception as e:
            logger.error(f"Text to speech error: {str(e)}")
            raise RuntimeError(f"TTS processing failed: {str(e)}")

    async def _convert_webm_to_wav(self, webm_data: bytes) -> bytes:
        """Convert WebM audio to WAV format"""
        try:
            # Use ffmpeg for conversion
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
                webm_file.write(webm_data)
                webm_path = webm_file.name
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                wav_path = wav_file.name
            
            try:
                # Convert using ffmpeg
                subprocess.run([
                    "ffmpeg", "-i", webm_path, "-ar", "16000", "-ac", "1", 
                    "-f", "wav", wav_path, "-y"
                ], check=True, capture_output=True)
                
                # Read converted WAV data
                with open(wav_path, "rb") as f:
                    wav_data = f.read()
                
                return wav_data
                
            finally:
                # Clean up temporary files
                if os.path.exists(webm_path):
                    os.unlink(webm_path)
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
                    
        except Exception as e:
            logger.error(f"Audio conversion error: {str(e)}")
            # Return original data if conversion fails
            return webm_data

    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calculate confidence score from Whisper result"""
        try:
            segments = whisper_result.get("segments", [])
            if not segments:
                return 0.0
            
            # Average confidence from segments
            total_confidence = sum(
                segment.get("avg_logprob", 0) for segment in segments
            )
            avg_confidence = total_confidence / len(segments)
            
            # Convert log probability to confidence (0-1)
            confidence = max(0.0, min(1.0, (avg_confidence + 1.0) / 2.0))
            return confidence
            
        except Exception:
            return 0.5  # Default confidence

    async def process_voice_interaction(
        self,
        user_id: str,
        agent_id: str,
        audio_data: bytes,
        format: str = "webm"
    ) -> Dict[str, Any]:
        """Process complete voice interaction"""
        try:
            # Step 1: Speech to Text
            stt_result = await self.speech_to_text(
                audio_data=audio_data,
                format=format,
                user_id=user_id
            )
            
            # Step 2: Get agent response (would be handled by LLM service)
            # This is a placeholder - actual implementation would call the LLM
            response_text = f"I heard you say: {stt_result['text']}"
            
            # Step 3: Text to Speech
            tts_result = await self.text_to_speech(
                text=response_text,
                user_id=user_id,
                agent_id=agent_id
            )
            
            return {
                "transcription": stt_result["text"],
                "confidence": stt_result["confidence"],
                "response": {
                    "text": response_text,
                    "audio": tts_result["audio_base64"]
                },
                "processing_time": tts_result["processing_time"]
            }
            
        except Exception as e:
            logger.error(f"Voice interaction error: {str(e)}")
            raise RuntimeError(f"Voice interaction failed: {str(e)}")

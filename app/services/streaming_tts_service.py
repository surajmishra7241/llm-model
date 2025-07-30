# app/services/streaming_tts_service.py
import asyncio
import logging
import base64
import io
import time
from typing import Dict, Any, Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# TTS imports
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("Coqui TTS not available")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

logger = logging.getLogger(__name__)

class StreamingTTSService:
    def __init__(self):
        self.tts_model = None
        self.pyttsx3_engine = None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")
        self.initialized = False
        self.chunk_size = 4096  # Audio chunk size for streaming
        
    async def initialize(self):
        """Initialize TTS models"""
        if self.initialized:
            return
            
        try:
            loop = asyncio.get_event_loop()
            
            if TTS_AVAILABLE:
                # Use Coqui TTS for better quality
                self.tts_model = await loop.run_in_executor(
                    self.executor,
                    lambda: TTS(model_name="tts_models/en/ljspeech/glow-tts")
                )
                logger.info("Initialized Coqui TTS")
            elif PYTTSX3_AVAILABLE:
                # Fallback to pyttsx3
                self.pyttsx3_engine = await loop.run_in_executor(
                    self.executor,
                    self._init_pyttsx3
                )
                logger.info("Initialized pyttsx3 TTS")
            else:
                raise Exception("No TTS engine available")
                
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise

    def _init_pyttsx3(self):
        """Initialize pyttsx3 engine"""
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)  # Speaking rate
        engine.setProperty('volume', 0.9)  # Volume
        
        # Set voice if available
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        
        return engine

    async def text_to_speech_stream(
        self, 
        text: str, 
        voice_config: Dict[str, Any] = None,
        session_id: str = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Convert text to speech and yield audio chunks"""
        
        if not self.initialized:
            await self.initialize()

        try:
            voice_config = voice_config or {}
            
            # Generate audio
            if self.tts_model:
                audio_data = await self._generate_with_coqui(text, voice_config)
            elif self.pyttsx3_engine:
                audio_data = await self._generate_with_pyttsx3(text, voice_config)
            else:
                raise Exception("No TTS engine available")

            # Stream audio in chunks
            async for chunk in self._stream_audio_chunks(audio_data, session_id):
                yield chunk

        except Exception as e:
            logger.error(f"TTS error: {e}")
            yield {
                'type': 'error',
                'error': str(e),
                'session_id': session_id
            }

    async def _generate_with_coqui(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        """Generate audio using Coqui TTS"""
        loop = asyncio.get_event_loop()
        
        def generate():
            try:
                # Generate audio to numpy array
                wav = self.tts_model.tts(text)
                
                # Convert to bytes (16-bit PCM)
                wav_int16 = (wav * 32767).astype(np.int16)
                return wav_int16.tobytes()
                
            except Exception as e:
                logger.error(f"Coqui TTS generation error: {e}")
                raise

        return await loop.run_in_executor(self.executor, generate)

    async def _generate_with_pyttsx3(self, text: str, voice_config: Dict[str, Any]) -> bytes:
        """Generate audio using pyttsx3"""
        loop = asyncio.get_event_loop()
        
        def generate():
            try:
                import tempfile
                import wave
                
                # Configure voice settings
                speaking_rate = voice_config.get('speaking_rate', 1.0)
                self.pyttsx3_engine.setProperty('rate', int(180 * speaking_rate))
                
                # Generate to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                self.pyttsx3_engine.save_to_file(text, temp_path)
                self.pyttsx3_engine.runAndWait()
                
                # Read the audio file
                with wave.open(temp_path, 'rb') as wav_file:
                    audio_data = wav_file.readframes(wav_file.getnframes())
                
                # Cleanup
                import os
                os.unlink(temp_path)
                
                return audio_data
                
            except Exception as e:
                logger.error(f"pyttsx3 TTS generation error: {e}")
                raise

        return await loop.run_in_executor(self.executor, generate)

    async def _stream_audio_chunks(
        self, 
        audio_data: bytes, 
        session_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream audio data in chunks"""
        try:
            total_chunks = len(audio_data) // self.chunk_size + (1 if len(audio_data) % self.chunk_size else 0)
            
            for i in range(0, len(audio_data), self.chunk_size):
                chunk = audio_data[i:i + self.chunk_size]
                chunk_base64 = base64.b64encode(chunk).decode('utf-8')
                
                chunk_info = {
                    'type': 'audio_chunk',
                    'audio_data': chunk_base64,
                    'chunk_index': i // self.chunk_size,
                    'total_chunks': total_chunks,
                    'format': 'pcm',
                    'sample_rate': 22050,
                    'channels': 1,
                    'session_id': session_id,
                    'is_final': (i + self.chunk_size) >= len(audio_data)
                }
                
                yield chunk_info
                
                # Small delay to prevent overwhelming the network
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error streaming audio chunks: {e}")
            yield {
                'type': 'error',
                'error': str(e),
                'session_id': session_id
            }

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.tts_model:
                del self.tts_model
            if self.pyttsx3_engine:
                del self.pyttsx3_engine
            self.executor.shutdown(wait=False)
            logger.info("TTS service cleaned up")
        except Exception as e:
            logger.error(f"TTS cleanup error: {e}")

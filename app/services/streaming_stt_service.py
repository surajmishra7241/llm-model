# app/services/streaming_stt_service.py
import asyncio
import logging
import base64
import io
import numpy as np
import torch
import whisper
import librosa
from typing import Dict, Any, Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import time

logger = logging.getLogger(__name__)

class StreamingSTTService:
    def __init__(self):
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt")
        self.audio_buffer = deque(maxlen=10)  # Store recent audio chunks
        self.overlap_duration = 0.5  # 500ms overlap
        self.chunk_duration = 2.0    # 2 second chunks
        self.sample_rate = 16000
        self.initialized = False
        
    async def initialize(self):
        """Initialize Whisper model"""
        if self.initialized:
            return
            
        try:
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                lambda: whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")
            )
            self.initialized = True
            logger.info("STT service initialized with Whisper")
        except Exception as e:
            logger.error(f"Failed to initialize STT: {e}")
            raise

    async def process_audio_chunk(self, audio_data: bytes, session_id: str) -> Dict[str, Any]:
        """Process streaming audio chunk with overlap handling"""
        try:
            # Decode audio from base64 if needed
            if isinstance(audio_data, str):
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data

            # Convert to numpy array
            audio_array = await self._bytes_to_audio_array(audio_bytes)
            
            # Add to buffer for overlap processing
            self.audio_buffer.append({
                'audio': audio_array,
                'timestamp': time.time(),
                'session_id': session_id
            })

            # Create overlapping segment
            combined_audio = await self._create_overlapping_segment()
            
            if combined_audio is None or len(combined_audio) < self.sample_rate * 0.5:
                return {
                    'type': 'interim',
                    'text': '',
                    'confidence': 0.0,
                    'session_id': session_id
                }

            # Transcribe with Whisper
            result = await self._transcribe_audio(combined_audio)
            
            # Determine if this is interim or final
            is_final = len(combined_audio) > self.sample_rate * 1.5  # More than 1.5s of audio
            
            return {
                'type': 'final' if is_final else 'interim',
                'text': result.get('text', '').strip(),
                'confidence': 0.95,  # Whisper doesn't provide confidence
                'language': result.get('language', 'en'),
                'session_id': session_id,
                'processing_time': result.get('processing_time', 0)
            }

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {
                'type': 'error',
                'error': str(e),
                'session_id': session_id
            }

    async def _bytes_to_audio_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        loop = asyncio.get_event_loop()
        
        def convert():
            try:
                # Try different formats
                audio_io = io.BytesIO(audio_bytes)
                
                # Use librosa for robust audio loading
                audio, sr = librosa.load(audio_io, sr=self.sample_rate, mono=True)
                return audio
            except Exception as e:
                logger.warning(f"Librosa failed, trying raw PCM: {e}")
                # Fallback: assume raw PCM 16-bit
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                return audio

        return await loop.run_in_executor(self.executor, convert)

    async def _create_overlapping_segment(self) -> Optional[np.ndarray]:
        """Create overlapping audio segment from buffer"""
        if len(self.audio_buffer) < 1:
            return None

        try:
            # Combine recent chunks with overlap
            combined = []
            overlap_samples = int(self.overlap_duration * self.sample_rate)
            
            for i, chunk_data in enumerate(self.audio_buffer):
                audio = chunk_data['audio']
                if i == 0:
                    combined.append(audio)
                else:
                    # Add with overlap
                    if len(combined) > 0:
                        overlap_start = max(0, len(combined[-1]) - overlap_samples)
                        combined.append(audio)
                    else:
                        combined.append(audio)

            if combined:
                return np.concatenate(combined)
            return None

        except Exception as e:
            logger.error(f"Error creating overlapping segment: {e}")
            return None

    async def _transcribe_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio using Whisper"""
        if not self.initialized:
            await self.initialize()

        loop = asyncio.get_event_loop()
        start_time = time.time()

        def transcribe():
            try:
                # Ensure audio is the right format for Whisper
                if len(audio.shape) > 1:
                    audio_mono = audio.mean(axis=1)
                else:
                    audio_mono = audio

                # Whisper expects audio length of at most 30 seconds
                max_samples = 30 * self.sample_rate
                if len(audio_mono) > max_samples:
                    audio_mono = audio_mono[:max_samples]

                result = self.model.transcribe(
                    audio_mono,
                    language='en',  # Set to auto-detect if needed
                    task='transcribe',
                    fp16=torch.cuda.is_available()
                )
                
                return {
                    'text': result['text'],
                    'language': result.get('language', 'en'),
                    'processing_time': time.time() - start_time
                }

            except Exception as e:
                logger.error(f"Whisper transcription error: {e}")
                return {
                    'text': '',
                    'language': 'en',
                    'processing_time': time.time() - start_time,
                    'error': str(e)
                }

        return await loop.run_in_executor(self.executor, transcribe)

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.model:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.executor.shutdown(wait=False)
            logger.info("STT service cleaned up")
        except Exception as e:
            logger.error(f"STT cleanup error: {e}")

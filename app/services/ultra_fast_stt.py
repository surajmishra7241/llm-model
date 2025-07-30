# ultra_fast_stt.py - Ultra-fast Speech-to-Text with Whisper optimization
import asyncio
import time
import logging
import base64
import io
import numpy as np
from typing import Dict, Any, Optional
import torch
import whisper
from whisper import load_model
import librosa
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

class UltraFastSTT:
    """Ultra-fast STT with <300ms target processing time"""
    
    def __init__(self, model_size: str = 'base', language: str = 'auto', target_time: float = 0.3):
        self.model_size = model_size
        self.language = language
        self.target_time = target_time
        self.logger = logging.getLogger(__name__)
        
        # Performance optimizations
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Audio preprocessing settings
        self.sample_rate = 16000
        self.chunk_duration = 30  # seconds
        
        # Performance tracking
        self.processing_times = []
        self.success_rate = 100.0
        
        # Initialize model
        asyncio.create_task(self.initialize_model())
    
    async def initialize_model(self):
        """Initialize Whisper model with optimizations"""
        try:
            self.logger.info(f"Loading Whisper {self.model_size} model on {self.device}")
            
            # Load model with optimizations
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                self._load_model_sync
            )
            
            # Warm up model
            await self.warmup_model()
            
            self.logger.info("Whisper model initialized and warmed up")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
    
    def _load_model_sync(self):
        """Synchronous model loading"""
        model = load_model(
            self.model_size,
            device=self.device,
            download_root=None,
            in_memory=True
        )
        
        # Optimize for inference
        if self.device == "cuda":
            model = model.half()  # Use FP16 for speed
        
        model.eval()
        return model
    
    async def warmup_model(self):
        """Warm up model with dummy audio"""
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        
        try:
            await self.transcribe_audio_array(dummy_audio)
            self.logger.info("Model warmup completed")
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")
    
    async def transcribe_async(self, audio_content: str, audio_format: str) -> Dict[str, Any]:
        """Asynchronous transcription with ultra-fast processing"""
        start_time = time.time()
        
        try:
            # Decode audio data
            audio_data = base64.b64decode(audio_content)
            
            # Convert to numpy array
            audio_array = await self.convert_audio_to_array(audio_data, audio_format)
            
            # Transcribe
            result = await self.transcribe_audio_array(audio_array)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Update success rate
            if processing_time <= self.target_time:
                self.success_rate = (self.success_rate * 0.9) + 10.0
            else:
                self.success_rate = (self.success_rate * 0.9) + 0.0
            
            self.logger.debug(f"STT completed in {processing_time:.3f}s (target: {self.target_time:.3f}s)")
            
            return {
                'text': result['text'].strip(),
                'confidence': self.calculate_confidence(result),
                'language': result.get('language', 'en'),
                'processing_time': processing_time,
                'within_target': processing_time <= self.target_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"STT failed in {processing_time:.3f}s: {e}")
            raise Exception(f"Speech-to-text failed: {str(e)}")
    
    async def convert_audio_to_array(self, audio_data: bytes, audio_format: str) -> np.ndarray:
        """Convert audio data to numpy array optimized for Whisper"""
        loop = asyncio.get_event_loop()
        
        def _convert():
            try:
                # Handle different audio formats
                if audio_format.lower() in ['webm', 'ogg']:
                    # Use librosa for WebM/OGG
                    audio_array, sr = librosa.load(
                        io.BytesIO(audio_data),
                        sr=self.sample_rate,
                        mono=True,
                        dtype=np.float32
                    )
                elif audio_format.lower() in ['wav', 'wave']:
                    # Use librosa for WAV
                    audio_array, sr = librosa.load(
                        io.BytesIO(audio_data),
                        sr=self.sample_rate,
                        mono=True,
                        dtype=np.float32
                    )
                elif audio_format.lower() == 'mp3':
                    # Use librosa for MP3
                    audio_array, sr = librosa.load(
                        io.BytesIO(audio_data),
                        sr=self.sample_rate,
                        mono=True,
                        dtype=np.float32
                    )
                else:
                    raise ValueError(f"Unsupported audio format: {audio_format}")
                
                # Normalize audio
                if len(audio_array) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                
                # Ensure minimum length
                if len(audio_array) < self.sample_rate * 0.1:  # Less than 100ms
                    raise ValueError("Audio too short for processing")
                
                # Limit max length for speed
                max_samples = self.sample_rate * self.chunk_duration
                if len(audio_array) > max_samples:
                    audio_array = audio_array[:max_samples]
                
                return audio_array
                
            except Exception as e:
                self.logger.error(f"Audio conversion error: {e}")
                raise
        
        return await loop.run_in_executor(self.executor, _convert)
    
    async def transcribe_audio_array(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Transcribe audio array with optimized Whisper"""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        loop = asyncio.get_event_loop()
        
        def _transcribe():
            try:
                # Optimize transcription options for speed
                options = {
                    'language': None if self.language == 'auto' else self.language,
                    'task': 'transcribe',
                    'fp16': self.device == 'cuda',
                    'beam_size': 1,  # Fastest beam search
                    'best_of': 1,    # No multiple samples
                    'temperature': [0.0],  # Deterministic
                    'compression_ratio_threshold': 2.4,
                    'logprob_threshold': -1.0,
                    'no_speech_threshold': 0.6,
                    'condition_on_previous_text': False  # Disable for speed
                }
                
                # Transcribe
                result = self.model.transcribe(
                    audio_array,
                    **options
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Whisper transcription error: {e}")
                raise
        
        return await loop.run_in_executor(self.executor, _transcribe)
    
    def calculate_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score from Whisper result"""
        try:
            # Extract confidence from segments
            segments = result.get('segments', [])
            if not segments:
                return 0.8  # Default confidence
            
            confidences = []
            for segment in segments:
                # Use average log probability as confidence proxy
                if 'avg_logprob' in segment:
                    # Convert log prob to confidence (0-1)
                    confidence = min(1.0, max(0.0, (segment['avg_logprob'] + 5) / 5))
                    confidences.append(confidence)
            
            if confidences:
                return sum(confidences) / len(confidences)
            else:
                return 0.8
                
        except Exception:
            return 0.8
    
    @lru_cache(maxsize=128)
    def get_language_info(self, language_code: str) -> Dict[str, str]:
        """Get language information with caching"""
        language_map = {
            'en': {'name': 'English', 'native': 'English'},
            'es': {'name': 'Spanish', 'native': 'Español'},
            'fr': {'name': 'French', 'native': 'Français'},
            'de': {'name': 'German', 'native': 'Deutsch'},
            'it': {'name': 'Italian', 'native': 'Italiano'},
            'pt': {'name': 'Portuguese', 'native': 'Português'},
            'ru': {'name': 'Russian', 'native': 'Русский'},
            'ja': {'name': 'Japanese', 'native': '日本語'},
            'ko': {'name': 'Korean', 'native': '한국어'},
            'zh': {'name': 'Chinese', 'native': '中文'},
        }
        
        return language_map.get(language_code, {'name': 'Unknown', 'native': 'Unknown'})
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.processing_times:
            return {
                'average_time': 0.0,
                'success_rate': 100.0,
                'target_achievement': 100.0,
                'total_requests': 0
            }
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        within_target = sum(1 for t in self.processing_times if t <= self.target_time)
        target_achievement = (within_target / len(self.processing_times)) * 100
        
        return {
            'average_time': avg_time,
            'success_rate': self.success_rate,
            'target_achievement': target_achievement,
            'total_requests': len(self.processing_times),
            'min_time': min(self.processing_times),
            'max_time': max(self.processing_times)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            self.executor.shutdown(wait=False)
            self.logger.info("STT cleanup completed")
            
        except Exception as e:
            self.logger.error(f"STT cleanup error: {e}")

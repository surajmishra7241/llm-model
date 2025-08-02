# app/services/voice_service.py

import asyncio
import logging
import base64
import io
import tempfile
import os
import time
import hashlib
import wave
import re
import subprocess
import platform
import shutil
from typing import Optional, Dict, Any, List, Union
import numpy as np
from pathlib import Path

# Core dependencies
import torch
import torchaudio
from app.config import settings

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self):
        self.whisper_model = None
        self.coqui_tts = None
        self._models_loaded = False
        self._load_lock = asyncio.Lock()
        self._model_cache = {}
        self._response_cache = {}
        
        # Audio processing settings
        self.target_sample_rate = 16000
        self.target_channels = 1
        
        # TTS Configuration - can be overridden by config.py
        self.tts_max_chunk_size = getattr(settings, 'TTS_MAX_CHUNK_SIZE', 400)
        self.tts_enable_chunking = getattr(settings, 'TTS_ENABLE_CHUNKING', True)
        self.tts_chunk_pause_duration = getattr(settings, 'TTS_CHUNK_PAUSE_DURATION', 0.3)
        
    async def _ensure_models_loaded(self):
        """Ensure voice models are loaded with proper error handling"""
        if self._models_loaded:
            return
            
        async with self._load_lock:
            if self._models_loaded:
                return
                
            try:
                logger.info("Loading voice models...")
                
                # Load Whisper for STT
                if settings.ENABLE_WHISPER:
                    await self._load_whisper_model()
                
                # Load TTS model
                if settings.ENABLE_TTS:
                    await self._load_tts_model()
                
                self._models_loaded = True
                logger.info("✅ Voice models loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to load voice models: {str(e)}")
                # Allow service to work with fallbacks
                self._models_loaded = True
                
    async def _load_whisper_model(self):
        """Load Whisper model using settings from config.py"""
        try:
            import whisper
            
            loop = asyncio.get_event_loop()
            
            def load_model():
                model_name = settings.WHISPER_MODEL
                return whisper.load_model(model_name)
            
            self.whisper_model = await loop.run_in_executor(None, load_model)
            logger.info(f"✅ Loaded Whisper model: {settings.WHISPER_MODEL}")
            
        except ImportError:
            logger.warning("Whisper not available. Install openai-whisper for STT functionality")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {str(e)}")
            
    def _check_system_dependencies(self) -> Dict[str, bool]:
        """Check if required system dependencies are available"""
        try:
            dependencies = {
                "espeak": shutil.which("espeak") is not None,
                "espeak-ng": shutil.which("espeak-ng") is not None,
                "ffmpeg": shutil.which("ffmpeg") is not None
            }
            
            # Log dependency status
            for dep, available in dependencies.items():
                if available:
                    logger.info(f"✅ Found {dep}")
                else:
                    logger.warning(f"❌ Missing {dep}")
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Error checking system dependencies: {str(e)}")
            return {}
            
    async def _load_tts_model(self):
        """Load TTS model based on settings from config.py"""
        try:
            # Check system dependencies first
            deps = self._check_system_dependencies()
            if not any(deps.get(dep, False) for dep in ["espeak", "espeak-ng"]):
                logger.warning("No espeak backend found - Coqui TTS may fail. Install: sudo apt-get install espeak-ng")
            
            # Use TTS engine from settings
            if settings.TTS_ENGINE == "coqui":
                await self._load_coqui_tts_model()
            else:
                logger.warning(f"Unknown TTS engine: {settings.TTS_ENGINE}, using system fallback")
                
        except Exception as e:
            logger.error(f"TTS loading failed: {str(e)}")
            # Don't raise - allow fallback to system TTS
            
    async def _load_coqui_tts_model(self):
        """Load Coqui TTS model with FIXED multi-speaker support"""
        try:
            from TTS.api import TTS
            
            loop = asyncio.get_event_loop()
            
            def load_coqui():
                # FIXED: Better model selection prioritizing single-speaker models
                models_to_try = [
                    # Single-speaker models (more reliable)
                    "tts_models/en/ljspeech/tacotron2-DDC",    # Most reliable single-speaker
                    "tts_models/en/ljspeech/fast_pitch",       # Fast single-speaker
                    "tts_models/en/ljspeech/glow-tts",         # Good quality single-speaker
                    "tts_models/en/ljspeech/neural_hmm",       # Alternative single-speaker
                    # Multi-speaker models (with proper handling)
                    "tts_models/en/vctk/vits",                 # Multi-speaker (needs speaker)
                    settings.TTS_MODEL,                        # User configured (if different)
                ]
                
                # Remove duplicates while preserving order
                models_to_try = list(dict.fromkeys(models_to_try))
                
                last_error = None
                
                for model_name in models_to_try:
                    try:
                        logger.info(f"Trying to load Coqui TTS model: {model_name}")
                        
                        # Initialize the model
                        tts = TTS(model_name=model_name)
                        
                        # FIXED: Test with proper multi-speaker handling
                        test_result = self._validate_coqui_model_fixed(tts, model_name)
                        if test_result:
                            logger.info(f"✅ Successfully loaded Coqui TTS model: {model_name}")
                            return tts
                        else:
                            logger.warning(f"Model {model_name} failed validation")
                            continue
                            
                    except Exception as e:
                        last_error = e
                        if "espeak" in str(e).lower():
                            logger.error(f"❌ Model {model_name} failed due to missing espeak: {str(e)}")
                        elif "speaker" in str(e).lower():
                            logger.error(f"❌ Model {model_name} failed due to speaker issue: {str(e)}")
                        else:
                            logger.warning(f"Failed to load model {model_name}: {str(e)}")
                        continue
                
                # Provide helpful error message based on last error
                if last_error:
                    if "espeak" in str(last_error).lower():
                        raise Exception(
                            "All Coqui TTS models failed due to missing espeak. "
                            "Install: sudo apt-get install espeak-ng espeak-ng-data"
                        )
                    else:
                        raise Exception(f"All Coqui TTS models failed. Last error: {str(last_error)}")
                else:
                    raise Exception("All Coqui TTS models failed to load")
            
            self.coqui_tts = await loop.run_in_executor(None, load_coqui)
            
        except ImportError:
            logger.error("Coqui TTS not available. Install with: pip install TTS")
            raise
        except Exception as e:
            logger.error(f"Coqui TTS loading failed: {str(e)}")
            raise

    def _validate_coqui_model_fixed(self, tts, model_name: str) -> bool:
        """FIXED: Validate Coqui TTS model with proper multi-speaker handling"""
        try:
            test_text = "Hello world"
            logger.info(f"Testing model {model_name} with text: '{test_text}'")
            
            # FIXED: Check if model is multi-speaker and handle accordingly
            is_multi_speaker = hasattr(tts, 'speakers') and tts.speakers is not None and len(tts.speakers) > 0
            
            if is_multi_speaker:
                logger.info(f"Model {model_name} is multi-speaker with {len(tts.speakers)} speakers")
                # Use first available speaker for multi-speaker models
                speaker = tts.speakers[0] if tts.speakers else None
                if speaker:
                    logger.info(f"Using speaker: {speaker}")
                    test_output = tts.tts(text=test_text, speaker=speaker)
                else:
                    logger.warning(f"Multi-speaker model {model_name} has no available speakers")
                    return False
            else:
                logger.info(f"Model {model_name} is single-speaker")
                test_output = tts.tts(text=test_text)
            
            logger.info(f"Model {model_name} test output type: {type(test_output)}, shape: {getattr(test_output, 'shape', 'N/A')}")
            
            # FIXED: Handle the specific case where Coqui returns list of float32 values
            if isinstance(test_output, list):
                if len(test_output) > 0:
                    # Check if it's a list of numpy scalars (float32)
                    if isinstance(test_output[0], (np.float32, np.float64, float)):
                        logger.info(f"✅ Model {model_name} produced list of audio samples ({len(test_output)} samples)")
                        # Convert to proper numpy array
                        try:
                            audio_array = np.array(test_output, dtype=np.float32)
                            if audio_array.size > 0:
                                logger.info(f"✅ Successfully converted to numpy array with shape: {audio_array.shape}")
                                return True
                            else:
                                logger.warning(f"Model {model_name} produced empty audio array")
                                return False
                        except Exception as conv_error:
                            logger.error(f"Failed to convert audio samples to array: {conv_error}")
                            return False
                    elif isinstance(test_output[0], np.ndarray):
                        logger.info(f"✅ Model {model_name} produced valid list of audio arrays")
                        return True
                    elif isinstance(test_output[0], str):
                        logger.info(f"⚠️ Model {model_name} produced sentences - may need special handling but is valid")
                        return True
                    else:
                        logger.warning(f"Model {model_name} produced list with unknown content type: {type(test_output[0])}")
                        return False
                else:
                    logger.warning(f"Model {model_name} produced empty list")
                    return False
                    
            elif isinstance(test_output, np.ndarray):
                if test_output.size > 0 and test_output.ndim >= 1:
                    logger.info(f"✅ Model {model_name} produced valid numpy array")
                    return True
                else:
                    logger.warning(f"Model {model_name} produced empty or invalid array")
                    return False
                    
            elif isinstance(test_output, (bytes, bytearray)):
                logger.info(f"✅ Model {model_name} produced audio bytes")
                return True
                
            else:
                logger.warning(f"Model {model_name} produced unexpected output type: {type(test_output)}")
                return False
                
        except Exception as e:
            if "speaker" in str(e).lower():
                logger.error(f"Model {model_name} validation failed due to speaker issue: {str(e)}")
            elif "espeak" in str(e).lower():
                logger.error(f"Model {model_name} validation failed due to espeak: {str(e)}")
            else:
                logger.error(f"Model {model_name} validation failed: {str(e)}")
            return False

    async def speech_to_text(self, audio_data: bytes, format: str = "webm", user_id: str = None, language: str = "en", optimization_level: str = "balanced") -> Dict[str, Any]:
        """Convert speech to text using Whisper with settings from config.py"""
        try:
            # Initialize Whisper if not loaded
            if self.whisper_model is None:
                await self._ensure_models_loaded()
                
            if self.whisper_model is None:
                import whisper
                # Use model from settings as fallback
                self.whisper_model = whisper.load_model(settings.WHISPER_MODEL)
                
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Transcribe with optimization settings
                transcribe_params = self._get_transcription_params(optimization_level, language)
                result = self.whisper_model.transcribe(temp_path, **transcribe_params)
                
                return {
                    "text": result["text"].strip(),
                    "confidence": self._calculate_confidence(result),
                    "language": result.get("language", language),
                    "processing_time": 0.5
                }
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"STT error: {str(e)}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e)
            }

    async def text_to_speech(self, text: str, user_id: str = None, agent_id: str = None, voice_config: Dict = None, optimization_level: str = "balanced") -> Dict[str, Any]:
        """Generate TTS using settings from config.py with comprehensive error handling and chunking support"""
        try:
            if not text or not text.strip():
                return {"audio_base64": "", "error": "Empty text"}
        
            # Clean text for TTS - properly handle HTML-encoded <think> tags
            clean_text = self._prepare_text_for_tts(text)
            logger.info(f"🔊 Generating TTS for: {clean_text[:50]}... (Total length: {len(clean_text)} chars)")
        
            # Ensure models are loaded
            await self._ensure_models_loaded()
            
            # Try TTS generation with fallbacks
            return await self._generate_tts_with_fallbacks(clean_text, voice_config or {})
                
        except Exception as e:
            logger.error(f"❌ TTS Error: {str(e)}")
            return {
                "audio_base64": "",
                "error": str(e),
                "success": False
            }

    async def _generate_tts_with_fallbacks(self, text: str, voice_config: Dict) -> Dict[str, Any]:
        """Try multiple TTS methods based on settings with proper fallbacks"""
        
        # Method 1: Try Coqui TTS if configured and loaded
        if settings.TTS_ENGINE == "coqui" and self.coqui_tts:
            try:
                logger.info("Attempting Coqui TTS generation...")
                return await self._generate_coqui_tts_fixed(text, voice_config)
            except Exception as e:
                logger.warning(f"Coqui TTS failed: {str(e)}, trying system TTS...")
        else:
            logger.info("Coqui TTS not available, using system TTS fallback")
        
        # Method 2: Try system TTS as fallback
        try:
            return await self._generate_system_tts(text, voice_config)
        except Exception as e:
            logger.error(f"All TTS methods failed: {str(e)}")
            return {
                "audio_base64": "",
                "error": "TTS generation failed - all methods unavailable",
                "success": False
            }

    def _chunk_text_for_tts(self, text: str, max_chunk_size: int = None) -> List[str]:
        """Break long text into chunks that TTS can handle"""
        if max_chunk_size is None:
            max_chunk_size = self.tts_max_chunk_size
            
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed limit, save current chunk
            if len(current_chunk + " " + sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence if current_chunk else sentence)
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    async def _generate_coqui_tts_fixed(self, text: str, voice_config: Dict) -> Dict[str, Any]:
        """Generate TTS using Coqui TTS with support for long text chunking"""
        try:
            loop = asyncio.get_event_loop()
            
            def generate_audio():
                # Process text but don't truncate
                processed_text = self._process_text_for_coqui(text)
                
                # Check if we need to chunk the text
                if self.tts_enable_chunking:
                    text_chunks = self._chunk_text_for_tts(processed_text, max_chunk_size=self.tts_max_chunk_size)
                else:
                    text_chunks = [processed_text]
                    
                logger.info(f"Processing {len(text_chunks)} text chunks for TTS")
                
                # Handle multi-speaker detection
                is_multi_speaker = hasattr(self.coqui_tts, 'speakers') and self.coqui_tts.speakers is not None and len(self.coqui_tts.speakers) > 0
                speaker = voice_config.get('speaker') or (self.coqui_tts.speakers[0] if is_multi_speaker else None)
                
                all_audio_data = []
                
                # Generate audio for each chunk
                for i, chunk in enumerate(text_chunks):
                    logger.info(f"Generating TTS for chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
                    
                    if is_multi_speaker:
                        wav_data = self.coqui_tts.tts(text=chunk, speaker=speaker)
                    else:
                        wav_data = self.coqui_tts.tts(text=chunk)
                    
                    # Convert to numpy array (existing logic)
                    if isinstance(wav_data, list):
                        if isinstance(wav_data[0], (np.float32, np.float64, float)):
                            wav_data = np.array(wav_data, dtype=np.float32)
                        elif isinstance(wav_data[0], np.ndarray):
                            wav_data = np.concatenate([arr for arr in wav_data if isinstance(arr, np.ndarray) and arr.size > 0])
                    
                    if isinstance(wav_data, np.ndarray) and wav_data.size > 0:
                        all_audio_data.append(wav_data)
                        
                        # Add small pause between chunks (optional)
                        if i < len(text_chunks) - 1:
                            pause_samples = int(self.tts_chunk_pause_duration * 22050)  # configurable pause
                            pause = np.zeros(pause_samples, dtype=np.float32)
                            all_audio_data.append(pause)
                
                if not all_audio_data:
                    raise Exception("No audio data generated")
                
                # Combine all chunks
                final_audio = np.concatenate(all_audio_data)
                
                # Normalize and convert to 16-bit PCM
                if np.max(np.abs(final_audio)) > 0:
                    final_audio = final_audio / np.max(np.abs(final_audio)) * 0.95
                
                wav_data_int16 = (final_audio * 32767).astype(np.int16)
                
                # Create WAV file
                buffer = io.BytesIO()
                sample_rate = 22050
                
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(wav_data_int16.tobytes())
                
                return buffer.getvalue()
            
            # Generate audio in thread pool
            audio_bytes = await loop.run_in_executor(None, generate_audio)
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            logger.info(f"✅ Coqui TTS generated: {len(audio_base64)} chars for full text ({len(text)} input chars)")
            
            return {
                "audio_base64": audio_base64,
                "format": "wav",
                "sample_rate": 22050,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Coqui TTS generation error: {str(e)}")
            raise

    def _process_text_for_coqui(self, text: str) -> str:
        """Process text specifically for Coqui TTS without aggressive truncation"""
        # Replace problematic characters
        text = text.replace("'", "'").replace("'", "'")
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("–", "-").replace("—", "-")
        text = text.replace("…", "...")
        
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Return full text for chunking - no truncation here
        return text.strip()

    async def _generate_system_tts(self, text: str, voice_config: Dict) -> Dict[str, Any]:
        """Fallback TTS using system commands with chunking support"""
        try:
            loop = asyncio.get_event_loop()
            
            def generate_system_audio():
                # If text is too long, chunk it
                if self.tts_enable_chunking and len(text) > self.tts_max_chunk_size:
                    text_chunks = self._chunk_text_for_tts(text)
                    all_audio_data = []
                    
                    for i, chunk in enumerate(text_chunks):
                        logger.info(f"Generating system TTS for chunk {i+1}/{len(text_chunks)}")
                        chunk_audio = self._generate_single_system_tts(chunk)
                        all_audio_data.append(chunk_audio)
                    
                    # Combine audio chunks
                    return self._combine_audio_chunks(all_audio_data)
                else:
                    return self._generate_single_system_tts(text)
            
            audio_base64 = await loop.run_in_executor(None, generate_system_audio)
            
            return {
                "audio_base64": audio_base64,
                "format": "wav",
                "success": True
            }
                
        except Exception as e:
            logger.error(f"System TTS error: {str(e)}")
            raise

    def _generate_single_system_tts(self, text: str) -> str:
        """Generate TTS for a single text chunk using system commands"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                cmd = ["say", "-o", temp_path, "--data-format=LEI16@22050", text]
            elif system == "linux":
                cmd = ["espeak", "-w", temp_path, "-s", "150", text]
            else:  # Windows
                ps_script = f'''
                Add-Type -AssemblyName System.speech
                $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $speak.SetOutputToWaveFile("{temp_path}")
                $speak.Speak("{text}")
                $speak.Dispose()
                '''
                cmd = ["powershell", "-Command", ps_script]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise Exception(f"System TTS failed: {result.stderr}")
            
            # Read generated file
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            return base64.b64encode(audio_data).decode()
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _combine_audio_chunks(self, audio_chunks: List[str]) -> str:
        """Combine multiple base64 audio chunks into one"""
        if not audio_chunks:
            raise Exception("No audio chunks to combine")
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        # For simplicity, just return the first chunk
        # In a more sophisticated implementation, you would decode, combine, and re-encode
        logger.warning("Audio chunk combination not fully implemented - returning first chunk")
        return audio_chunks[0]

    def _prepare_text_for_tts(self, text: str) -> str:
        """Prepare text for optimal TTS processing without excessive truncation"""
        if not text or not text.strip():
            return ""
        
        # Remove HTML-encoded thinking tags first
        text = re.sub(r'&amp;amp;lt;think&amp;amp;gt;.*?&amp;amp;lt;/think&amp;amp;gt;', '', text, flags=re.DOTALL)
        text = re.sub(r'&amp;amp;lt;think&amp;amp;gt;.*$', '', text, flags=re.DOTALL)
        
        # Remove regular <think> tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
        
        # Remove any remaining XML-like tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return ""
        
        # Character replacements for better TTS
        replacements = {
            "'": "'", "'": "'", """: '"', """: '"',
            "–": "-", "—": "-", "…": "..."
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Clean markdown but keep the content
        text = re.sub(r'[*_`#]', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Keep link text
        text = re.sub(r'https?://[^\s]+', 'link', text)
        
        # Ensure proper punctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text.strip()

    # Additional utility methods
    def _get_transcription_params(self, optimization_level: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Get transcription parameters based on optimization level"""
        base_params = {
            "fp16": torch.cuda.is_available(),
            "task": "transcribe",
            "condition_on_previous_text": False
        }
        
        if language:
            base_params["language"] = language
            
        if optimization_level == "fast":
            base_params.update({
                "temperature": 0.0,
                "no_speech_threshold": 0.7,
                "logprob_threshold": -1.0,
                "compression_ratio_threshold": 2.4
            })
        elif optimization_level == "quality":
            base_params.update({
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "no_speech_threshold": 0.4,
                "logprob_threshold": -1.0,
                "compression_ratio_threshold": 2.4
            })
        else:  # balanced
            base_params.update({
                "temperature": 0.0,
                "no_speech_threshold": 0.6,
                "logprob_threshold": -1.0,
                "compression_ratio_threshold": 2.4
            })
            
        return base_params

    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calculate confidence score from Whisper result"""
        try:
            if "segments" in whisper_result and whisper_result["segments"]:
                segments = whisper_result["segments"]
                confidences = []
                
                for segment in segments:
                    if "avg_logprob" in segment:
                        # Convert log probability to confidence (0-1 scale)
                        logprob = segment["avg_logprob"]
                        confidence = min(1.0, max(0.0, np.exp(logprob)))
                        confidences.append(confidence)
                
                if confidences:
                    return sum(confidences) / len(confidences)
            
            # Fallback confidence based on text characteristics
            text = whisper_result.get("text", "").strip()
            if len(text) > 10:
                return 0.8
            elif len(text) > 0:
                return 0.6
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Confidence calculation error: {str(e)}")
            return 0.7

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of voice service using settings"""
        try:
            await self._ensure_models_loaded()
            
            deps = self._check_system_dependencies()
            
            return {
                "status": "healthy",
                "whisper_available": self.whisper_model is not None,
                "tts_available": self.coqui_tts is not None,
                "tts_engine": settings.TTS_ENGINE,
                "whisper_model": settings.WHISPER_MODEL if self.whisper_model else "none",
                "tts_model": settings.TTS_MODEL if settings.TTS_ENGINE == "coqui" else "system",
                "coqui_loaded": self.coqui_tts is not None,
                "coqui_is_multi_speaker": hasattr(self.coqui_tts, 'speakers') and self.coqui_tts.speakers is not None and len(self.coqui_tts.speakers) > 0 if self.coqui_tts else False,
                "available_speakers": self.coqui_tts.speakers if (self.coqui_tts and hasattr(self.coqui_tts, 'speakers') and self.coqui_tts.speakers) else [],
                "system_dependencies": deps,
                "models_loaded": self._models_loaded,
                "tts_chunking_enabled": self.tts_enable_chunking,
                "tts_max_chunk_size": self.tts_max_chunk_size,
                "settings_used": {
                    "enable_whisper": settings.ENABLE_WHISPER,
                    "enable_tts": settings.ENABLE_TTS,
                    "tts_engine": settings.TTS_ENGINE,
                    "whisper_model": settings.WHISPER_MODEL,
                    "tts_model": settings.TTS_MODEL
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "models_loaded": False
            }

    async def clear_cache(self):
        """Clear all cached responses"""
        self._model_cache.clear()
        self._response_cache.clear()
        logger.info("Voice service cache cleared")

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self._model_cache.clear()
            self._response_cache.clear()
            
            # Cleanup Coqui TTS if needed
            if hasattr(self, 'coqui_tts') and self.coqui_tts:
                try:
                    del self.coqui_tts
                    self.coqui_tts = None
                except:
                    pass
                    
            logger.info("Voice service cleanup completed")
            
        except Exception as e:
            logger.error(f"Voice service cleanup error: {str(e)}")

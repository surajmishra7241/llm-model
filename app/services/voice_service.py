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
        """Generate TTS using settings from config.py with comprehensive error handling"""
        try:
            if not text or not text.strip():
                return {"audio_base64": "", "error": "Empty text"}
        
            # Clean text for TTS - properly handle HTML-encoded <think> tags
            clean_text = self._prepare_text_for_tts(text)
            logger.info(f"🔊 Generating TTS for: {clean_text[:50]}...")
        
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

    async def _generate_coqui_tts_fixed(self, text: str, voice_config: Dict) -> Dict[str, Any]:
        """Generate TTS using Coqui TTS - FIXED for multi-speaker models and list output"""
        try:
            loop = asyncio.get_event_loop()
            
            def generate_audio():
                # Handle text preprocessing for Coqui
                processed_text = self._process_text_for_coqui(text)
                
                # FIXED: Check if model is multi-speaker and handle accordingly
                is_multi_speaker = hasattr(self.coqui_tts, 'speakers') and self.coqui_tts.speakers is not None and len(self.coqui_tts.speakers) > 0
                
                logger.info(f"Generating Coqui TTS for: {processed_text[:100]}...")
                
                if is_multi_speaker:
                    # Use specified speaker or first available speaker
                    speaker = voice_config.get('speaker') or self.coqui_tts.speakers[0]
                    logger.info(f"Using multi-speaker model with speaker: {speaker}")
                    wav_data = self.coqui_tts.tts(text=processed_text, speaker=speaker)
                else:
                    logger.info("Using single-speaker model")
                    wav_data = self.coqui_tts.tts(text=processed_text)
                
                logger.info(f"Coqui TTS output type: {type(wav_data)}, shape: {getattr(wav_data, 'shape', 'N/A')}")
                
                # FIXED: Handle different return types including list of float32 values
                if isinstance(wav_data, list):
                    # Handle list of audio samples or arrays
                    if len(wav_data) == 0:
                        raise Exception("Coqui TTS returned empty list")
                    
                    if isinstance(wav_data[0], str):
                        # Rejoin sentences and try again
                        combined_text = ' '.join(wav_data)
                        logger.info(f"Coqui returned sentences, rejoining: {len(wav_data)} parts")
                        wav_data = self.coqui_tts.tts(text=combined_text, speaker=speaker if is_multi_speaker else None)
                        
                        # Process the rejoined result
                        if isinstance(wav_data, list) and len(wav_data) > 0 and isinstance(wav_data[0], np.ndarray):
                            wav_data = np.concatenate([arr for arr in wav_data if isinstance(arr, np.ndarray) and arr.size > 0])
                        elif not isinstance(wav_data, np.ndarray):
                            raise Exception("Failed to get audio data after sentence rejoining")
                            
                    elif isinstance(wav_data[0], np.ndarray):
                        # Concatenate audio arrays, filtering out empty ones
                        valid_arrays = [arr for arr in wav_data if isinstance(arr, np.ndarray) and arr.size > 0]
                        if not valid_arrays:
                            raise Exception("All audio arrays are empty")
                        wav_data = np.concatenate(valid_arrays)
                        
                    elif isinstance(wav_data[0], (np.float32, np.float64, float)):
                        # FIXED: Handle list of float32/float64 audio samples (the main issue)
                        logger.info(f"Converting list of {len(wav_data)} audio samples to numpy array")
                        wav_data = np.array(wav_data, dtype=np.float32)
                        if wav_data.size == 0:
                            raise Exception("Converted audio array is empty")
                        logger.info(f"Successfully converted to array with shape: {wav_data.shape}")
                        
                    else:
                        raise Exception(f"Unexpected list content type: {type(wav_data[0])}")
                        
                elif isinstance(wav_data, np.ndarray):
                    # Check for zero-dimensional or empty arrays
                    if wav_data.size == 0:
                        raise Exception("Coqui TTS returned empty array")
                    if wav_data.ndim == 0:
                        raise Exception("Coqui TTS returned zero-dimensional array")
                    
                    # Ensure it's 1D
                    if wav_data.ndim > 1:
                        wav_data = wav_data.flatten()
                        
                elif isinstance(wav_data, (bytes, bytearray)):
                    # Already audio bytes, return directly
                    return wav_data
                else:
                    raise Exception(f"Unexpected Coqui TTS output type: {type(wav_data)}")
                
                # Validate final array
                if not isinstance(wav_data, np.ndarray) or wav_data.size == 0:
                    raise Exception("Invalid final audio array")
                
                # Normalize audio to prevent clipping
                if np.max(np.abs(wav_data)) > 0:
                    wav_data = wav_data / np.max(np.abs(wav_data)) * 0.95
                
                # Convert to 16-bit PCM
                wav_data_int16 = (wav_data * 32767).astype(np.int16)
                
                # Create WAV file in memory
                buffer = io.BytesIO()
                
                # Get sample rate from model or use default
                sample_rate = 22050
                if hasattr(self.coqui_tts, 'synthesizer'):
                    if hasattr(self.coqui_tts.synthesizer, 'output_sample_rate'):
                        sample_rate = self.coqui_tts.synthesizer.output_sample_rate
                
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(wav_data_int16.tobytes())
                
                return buffer.getvalue()
            
            # Generate audio in thread pool
            audio_bytes = await loop.run_in_executor(None, generate_audio)
            
            # Encode to base64
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            logger.info(f"✅ Coqui TTS generated: {len(audio_base64)} chars")
            
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
        """Process text specifically for Coqui TTS to avoid vocabulary issues"""
        # Replace problematic characters that cause vocabulary warnings
        text = text.replace("'", "'")  # Replace curly apostrophe with straight
        text = text.replace("'", "'")  # Replace other curly apostrophe
        text = text.replace(""", '"')  # Replace curly quotes
        text = text.replace(""", '"')
        text = text.replace("–", "-")   # Replace en dash
        text = text.replace("—", "-")   # Replace em dash
        text = text.replace("…", "...")  # Replace ellipsis
        
        # Remove other special Unicode characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
        
        # Ensure text doesn't get split into problematic sentences
        if len(text) > 300:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 1:
                # Keep first few sentences that fit within limit
                result_sentences = []
                char_count = 0
                for sentence in sentences:
                    if char_count + len(sentence) <= 250:  # Conservative limit
                        result_sentences.append(sentence)
                        char_count += len(sentence)
                    else:
                        break
                
                if result_sentences:
                    text = ' '.join(result_sentences)
                    if not text.endswith(('.', '!', '?')):
                        text += '.'
                else:
                    text = text[:250] + '.'
        
        return text

    async def _generate_system_tts(self, text: str, voice_config: Dict) -> Dict[str, Any]:
        """Fallback TTS using system commands"""
        try:
            loop = asyncio.get_event_loop()
            
            def generate_system_audio():
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    system = platform.system().lower()
                    
                    if system == "darwin":  # macOS
                        # Use built-in say command (No objc issues)
                        cmd = ["say", "-o", temp_path, "--data-format=LEI16@22050", text]
                    elif system == "linux":
                        # Use espeak
                        cmd = ["espeak", "-w", temp_path, "-s", "150", text]
                    else:  # Windows
                        # Use PowerShell
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
            
            audio_base64 = await loop.run_in_executor(None, generate_system_audio)
            
            return {
                "audio_base64": audio_base64,
                "format": "wav",
                "success": True
            }
                
        except Exception as e:
            logger.error(f"System TTS error: {str(e)}")
            raise

    def _prepare_text_for_tts(self, text: str) -> str:
        """Prepare text for optimal TTS processing - FIXED to handle HTML-encoded <think> tags"""
        if not text or not text.strip():
            return ""
        
        # FIXED: Remove HTML-encoded thinking tags first (from your error logs)
        text = re.sub(r'&amp;lt;think&amp;gt;.*?&amp;lt;/think&amp;gt;', '', text, flags=re.DOTALL)
        text = re.sub(r'&amp;lt;think&amp;gt;.*$', '', text, flags=re.DOTALL)
        
        # Remove regular <think> tags
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
        
        # Remove any remaining XML-like tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If no text remains after cleaning, return empty
        if not text:
            return ""
        
        # Clean problematic characters for TTS
        text = text.replace("'", "'")  # Replace curly apostrophe
        text = text.replace("'", "'")
        text = text.replace(""", '"')  # Replace curly quotes
        text = text.replace(""", '"')
        text = text.replace("–", "-")   # Replace en dash
        text = text.replace("—", "-")   # Replace em dash
        text = text.replace("…", "...")  # Replace ellipsis
        
        # Clean markdown and special characters
        text = re.sub(r'[*_`#\[\]()]', '', text)
        text = re.sub(r'https?://[^\s]+', 'link', text)  # Replace URLs
        
        # Ensure proper punctuation for natural speech
        if text and not text[-1] in '.!?':
            text += '.'
        
        # Handle abbreviations and numbers for better pronunciation
        replacements = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMrs\.': 'Missis',
            r'\bMs\.': 'Miss',
            r'\betc\.': 'etcetera',
            r'\bi\.e\.': 'that is',
            r'\be\.g\.': 'for example'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    # Additional utility methods (keeping existing implementations)
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

# app/utils/tts_model_manager.py
import logging
import numpy as np
from typing import Optional, Dict, Any, List
from TTS.api import TTS

logger = logging.getLogger(__name__)

class TTSModelManager:
    """Utility class to manage TTS model loading and validation"""
    
    RELIABLE_MODELS = [
        "tts_models/en/ljspeech/tacotron2-DDC",    # Most reliable single-speaker
        "tts_models/en/ljspeech/fast_pitch",       # Fast single-speaker
        "tts_models/en/ljspeech/glow-tts",         # Good quality single-speaker
        "tts_models/en/ljspeech/neural_hmm",       # Alternative single-speaker
    ]
    
    MULTI_SPEAKER_MODELS = [
        "tts_models/en/vctk/vits",                 # Multi-speaker
        "tts_models/multilingual/multi-dataset/your_tts",  # Multi-lingual multi-speaker
    ]
    
    @staticmethod
    def load_best_available_model(preferred_models: List[str] = None) -> Optional[TTS]:
        """Load the best available TTS model"""
        models_to_try = preferred_models or (TTSModelManager.RELIABLE_MODELS + TTSModelManager.MULTI_SPEAKER_MODELS)
        
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load TTS model: {model_name}")
                tts = TTS(model_name=model_name)
                
                if TTSModelManager.validate_model(tts, model_name):
                    logger.info(f"✅ Successfully loaded and validated: {model_name}")
                    return tts
                else:
                    logger.warning(f"Model {model_name} failed validation")
                    
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
                continue
        
        logger.error("No TTS models could be loaded")
        return None
    
    @staticmethod
    def validate_model(tts: TTS, model_name: str) -> bool:
        """Validate that a TTS model works correctly"""
        try:
            test_text = "Hello world"
            is_multi_speaker = hasattr(tts, 'speakers') and tts.speakers and len(tts.speakers) > 0
            
            # Generate test audio
            if is_multi_speaker:
                speaker = tts.speakers[0]
                wav_data = tts.tts(text=test_text, speaker=speaker)
            else:
                wav_data = tts.tts(text=test_text)
            
            # Validate output
            return TTSModelManager._validate_audio_output(wav_data, model_name)
            
        except Exception as e:
            logger.error(f"Model validation failed for {model_name}: {str(e)}")
            return False
    
    @staticmethod
    def _validate_audio_output(wav_data: Any, model_name: str) -> bool:
        """Validate the audio output from TTS"""
        try:
            # Handle list of float32 values (common Coqui output)
            if isinstance(wav_data, list):
                if len(wav_data) == 0:
                    return False
                
                # Check if it's a list of audio samples
                if isinstance(wav_data[0], (np.float32, np.float64, float)):
                    # Convert to numpy array
                    audio_array = np.array(wav_data, dtype=np.float32)
                    return audio_array.size > 0
                
                # Check if it's a list of numpy arrays
                elif isinstance(wav_data[0], np.ndarray):
                    return all(arr.size > 0 for arr in wav_data)
                
                # Check if it's a list of sentences (needs rejoining)
                elif isinstance(wav_data[0], str):
                    return True  # Valid but needs special handling
                
            # Handle numpy arrays
            elif isinstance(wav_data, np.ndarray):
                return wav_data.size > 0 and wav_data.ndim >= 1
            
            # Handle bytes
            elif isinstance(wav_data, (bytes, bytearray)):
                return len(wav_data) > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Audio validation error: {str(e)}")
            return False
    
    @staticmethod  
    def convert_to_numpy_array(wav_data: Any) -> np.ndarray:
        """Convert various TTS output formats to numpy array"""
        if isinstance(wav_data, list):
            if len(wav_data) == 0:
                raise ValueError("Empty audio data")
            
            # Handle list of float32 values
            if isinstance(wav_data[0], (np.float32, np.float64, float)):
                return np.array(wav_data, dtype=np.float32)
            
            # Handle list of numpy arrays
            elif isinstance(wav_data[0], np.ndarray):
                valid_arrays = [arr for arr in wav_data if arr.size > 0]
                if not valid_arrays:
                    raise ValueError("No valid audio arrays")
                return np.concatenate(valid_arrays)
            
            else:
                raise ValueError(f"Unsupported list content type: {type(wav_data[0])}")
        
        elif isinstance(wav_data, np.ndarray):
            if wav_data.size == 0:
                raise ValueError("Empty numpy array")
            if wav_data.ndim > 1:
                wav_data = wav_data.flatten()
            return wav_data.astype(np.float32)
        
        else:
            raise ValueError(f"Unsupported audio data type: {type(wav_data)}")

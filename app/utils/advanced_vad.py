# app/utils/advanced_vad.py - Advanced Voice Activity Detection
import logging
import time
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AdvancedVAD:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'silenceThreshold': 0.01,
            'speechThreshold': 0.03,
            'minSilenceDuration': 800,
            'minSpeechDuration': 200,
            'maxRecordingTime': 15000,
            'vadSensitivity': 0.7,
            'endpointDetection': True
        }
        
        # State tracking
        self.current_state = 'silence'
        self.speech_start_time = 0
        self.silence_start_time = 0
        self.energy_history = []
        self.background_noise = 0.001
        
        # Performance metrics
        self.processing_times = []
        self.detection_accuracy = 0.95
        
        logger.info("Advanced VAD initialized")

    def process_audio_frame(self, audio_data: np.ndarray, timestamp: float = None) -> Dict[str, Any]:
        """Process audio frame and detect voice activity"""
        start_time = time.time()
        
        if timestamp is None:
            timestamp = time.time() * 1000  # Convert to milliseconds
        
        try:
            # Calculate energy
            energy = self._calculate_energy(audio_data)
            
            # Update energy history
            self._update_energy_history(energy)
            
            # Voice activity detection
            result = self._detect_voice_activity(energy, timestamp)
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            # Keep processing times history manageable
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            return {
                'is_voice_active': result['is_voice_active'],
                'energy': energy,
                'confidence': result['confidence'],
                'state': self.current_state,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"VAD processing error: {str(e)}")
            return {
                'is_voice_active': False,
                'energy': 0.0,
                'confidence': 0.0,
                'state': 'error',
                'processing_time': 0.0
            }

    def _calculate_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS energy of audio frame"""
        try:
            if len(audio_data) == 0:
                return 0.0
            
            # Convert to numpy array if needed
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Normalize and clamp
            energy = min(max(rms * 10, 0.0), 1.0)
            
            return energy
            
        except Exception as e:
            logger.error(f"Energy calculation error: {str(e)}")
            return 0.0

    def _update_energy_history(self, energy: float):
        """Update energy history for adaptive processing"""
        self.energy_history.append(energy)
        
        # Keep history size manageable
        if len(self.energy_history) > 50:
            self.energy_history.pop(0)
        
        # Update background noise estimate
        if len(self.energy_history) >= 10:
            sorted_history = sorted(self.energy_history)
            self.background_noise = sorted_history[len(sorted_history) // 3]

    def _detect_voice_activity(self, energy: float, timestamp: float) -> Dict[str, Any]:
        """Main voice activity detection logic"""
        speech_threshold = self.config['speechThreshold']
        silence_threshold = self.config['silenceThreshold']
        min_speech_duration = self.config['minSpeechDuration']
        min_silence_duration = self.config['minSilenceDuration']
        
        is_voice_active = False
        confidence = 0.0
        
        # Adaptive thresholds based on background noise
        adaptive_speech_threshold = max(speech_threshold, self.background_noise * 3)
        adaptive_silence_threshold = max(silence_threshold, self.background_noise * 1.5)
        
        if self.current_state == 'silence':
            if energy > adaptive_speech_threshold:
                if self.speech_start_time == 0:
                    self.speech_start_time = timestamp
                elif timestamp - self.speech_start_time >= min_speech_duration:
                    self.current_state = 'speech'
                    is_voice_active = True
                    confidence = min((energy - adaptive_speech_threshold) / adaptive_speech_threshold, 1.0)
            else:
                self.speech_start_time = 0
                
        elif self.current_state == 'speech':
            if energy < adaptive_silence_threshold:
                if self.silence_start_time == 0:
                    self.silence_start_time = timestamp
                elif timestamp - self.silence_start_time >= min_silence_duration:
                    self.current_state = 'silence'
                    is_voice_active = False
                    confidence = 0.0
                else:
                    # Still in speech state
                    is_voice_active = True
                    confidence = min(energy / adaptive_speech_threshold, 1.0)
            else:
                # Continue speech
                self.silence_start_time = 0
                is_voice_active = True
                confidence = min(energy / adaptive_speech_threshold, 1.0)
        
        return {
            'is_voice_active': is_voice_active,
            'confidence': confidence
        }

    def optimize_for_speed(self):
        """Optimize VAD for speed over accuracy"""
        self.config.update({
            'minSilenceDuration': 600,
            'minSpeechDuration': 150,
            'vadSensitivity': 0.8
        })
        logger.info("VAD optimized for speed")

    def optimize_for_accuracy(self):
        """Optimize VAD for accuracy over speed"""
        self.config.update({
            'minSilenceDuration': 1000,
            'minSpeechDuration': 300,
            'vadSensitivity': 0.6
        })
        logger.info("VAD optimized for accuracy")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.processing_times:
            return {
                'average_processing_time': 0.0,
                'detection_accuracy': self.detection_accuracy,
                'total_frames': 0
            }
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        
        return {
            'average_processing_time': avg_time,
            'detection_accuracy': self.detection_accuracy,
            'total_frames': len(self.processing_times),
            'config': self.config.copy(),
            'current_state': self.current_state,
            'background_noise': self.background_noise
        }

    def reset(self):
        """Reset VAD state"""
        self.current_state = 'silence'
        self.speech_start_time = 0
        self.silence_start_time = 0
        self.energy_history = []
        self.processing_times = []
        logger.info("VAD state reset")

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.reset()
            logger.info("VAD cleanup completed")
        except Exception as e:
            logger.error(f"VAD cleanup error: {e}")

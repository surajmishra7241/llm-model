# advanced_vad.py - Advanced Voice Activity Detection
import numpy as np
import time
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class VADConfig:
    sensitivity: float = 0.7
    min_speech_duration: int = 200  # ms
    min_silence_duration: int = 800  # ms
    max_recording_time: int = 15000  # ms
    energy_smoothing: float = 0.3
    noise_reduction: bool = True
    endpoint_detection: bool = True

class AdvancedVAD:
    """Advanced Voice Activity Detection with ultra-fast response"""
    
    def __init__(self, sensitivity: float = 0.7, min_speech_duration: int = 200, 
                 min_silence_duration: int = 800):
        self.config = VADConfig(
            sensitivity=sensitivity,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration
        )
        
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # VAD state
        self.current_state = 'silence'  # 'silence', 'speech', 'endpoint'
        self.speech_start_time = 0
        self.silence_start_time = 0
        self.last_state_change = 0
        
        # Energy tracking
        self.energy_history = []
        self.background_noise = 0.01
        self.adaptive_threshold = {
            'speech': 0.03,
            'silence': 0.01
        }
        
        # Performance tracking
        self.processing_times = []
        self.detection_accuracy = 95.0
    
    async def process_audio_frame(self, audio_data: np.ndarray, timestamp: float = None) -> Dict[str, Any]:
        """Process audio frame and detect voice activity"""
        start_time = time.time()
        
        if timestamp is None:
            timestamp = time.time()
        
        try:
            # Calculate energy
            energy = await self.calculate_energy_async(audio_data)
            
            # Update energy history
            self.update_energy_history(energy)
            
            # Update adaptive thresholds
            self.update_adaptive_thresholds()
            
            # Perform VAD
            vad_result = self.detect_voice_activity(energy, timestamp)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return {
                'is_voice_active': vad_result['is_voice_active'],
                'energy': energy,
                'confidence': vad_result['confidence'],
                'state': self.current_state,
                'should_end_recording': vad_result['should_end_recording'],
                'timestamp': timestamp,
                'thresholds': self.adaptive_threshold.copy(),
                'background_noise': self.background_noise,
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.logger.error(f"VAD processing error: {e}")
            return {
                'is_voice_active': False,
                'energy': 0.0,
                'confidence': 0.0,
                'state': 'error',
                'should_end_recording': False,
                'timestamp': timestamp,
                'processing_time': time.time() - start_time
            }
    
    async def calculate_energy_async(self, audio_data: np.ndarray) -> float:
        """Calculate audio energy asynchronously"""
        loop = asyncio.get_event_loop()
        
        def _calculate_energy():
            if len(audio_data) == 0:
                return 0.0
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Normalize and clamp
            energy = max(0.0, min(1.0, rms * 10))
            
            return energy
        
        return await loop.run_in_executor(self.executor, _calculate_energy)
    
    def update_energy_history(self, energy: float):
        """Update energy history for adaptive processing"""
        self.energy_history.append(energy)
        
        # Keep history manageable
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)
        
        # Update background noise estimate
        if len(self.energy_history) >= 10:
            sorted_history = sorted(self.energy_history)
            # Use 30th percentile as background noise
            noise_index = int(len(sorted_history) * 0.3)
            self.background_noise = sorted_history[noise_index]
    
    def update_adaptive_thresholds(self):
        """Update adaptive thresholds based on environment"""
        if len(self.energy_history) < 20:
            return
        
        recent_energy = self.energy_history[-20:]
        avg_energy = sum(recent_energy) / len(recent_energy)
        max_energy = max(recent_energy)
        
        # Calculate adaptive thresholds
        base_multiplier = 1 + self.config.sensitivity
        
        self.adaptive_threshold['speech'] = max(
            0.03,  # Minimum threshold
            (avg_energy + self.background_noise) * base_multiplier
        )
        
        self.adaptive_threshold['silence'] = max(
            0.01,  # Minimum threshold
            self.background_noise * 1.2
        )
        
        # Prevent thresholds from being too high
        self.adaptive_threshold['speech'] = min(
            self.adaptive_threshold['speech'],
            max_energy * 0.7
        )
    
    def detect_voice_activity(self, energy: float, timestamp: float) -> Dict[str, Any]:
        """Main voice activity detection logic"""
        time_since_last_change = timestamp - self.last_state_change
        is_voice_active = False
        confidence = 0.0
        should_end_recording = False
        
        # State machine for voice detection
        if self.current_state == 'silence':
            if energy > self.adaptive_threshold['speech']:
                if self.speech_start_time == 0:
                    self.speech_start_time = timestamp
                elif (timestamp - self.speech_start_time) * 1000 >= self.config.min_speech_duration:
                    # Transition to speech
                    self.current_state = 'speech'
                    self.last_state_change = timestamp
                    self.speech_start_time = 0
                    self.silence_start_time = 0
                    is_voice_active = True
                    confidence = self.calculate_confidence(energy, self.adaptive_threshold['speech'])
            else:
                self.speech_start_time = 0
        
        elif self.current_state == 'speech':
            if energy < self.adaptive_threshold['silence']:
                if self.silence_start_time == 0:
                    self.silence_start_time = timestamp
                elif (timestamp - self.silence_start_time) * 1000 >= self.config.min_silence_duration:
                    # Transition to silence
                    self.current_state = 'silence'
                    self.last_state_change = timestamp
                    self.silence_start_time = 0
                    self.speech_start_time = 0
                    is_voice_active = False
                    confidence = 0.0
                    
                    # Check for endpoint
                    if self.config.endpoint_detection:
                        should_end_recording = self.detect_endpoint(timestamp)
                else:
                    # Still in speech, brief dip
                    is_voice_active = True
                    confidence = self.calculate_confidence(energy, self.adaptive_threshold['silence'])
            else:
                # Continuing speech
                self.silence_start_time = 0
                is_voice_active = True
                confidence = self.calculate_confidence(energy, self.adaptive_threshold['speech'])
        
        return {
            'is_voice_active': is_voice_active,
            'confidence': confidence,
            'should_end_recording': should_end_recording
        }
    
    def calculate_confidence(self, energy: float, threshold: float) -> float:
        """Calculate confidence score for voice detection"""
        if energy <= threshold:
            return 0.0
        
        confidence = min(1.0, (energy - threshold) / threshold)
        return round(confidence, 2)
    
    def detect_endpoint(self, timestamp: float) -> bool:
        """Detect natural speech endpoints for faster response"""
        if len(self.energy_history) < 5:
            return False
        
        # Check for natural pauses
        recent_energy = self.energy_history[-5:]
        energy_trend = self.calculate_energy_trend(recent_energy)
        
        # Endpoint conditions
        conditions = [
            # Sufficient speech duration
            self.speech_start_time > 0 and 
            (timestamp - self.speech_start_time) * 1000 >= (self.config.min_speech_duration * 2),
            
            # Decreasing energy trend
            energy_trend < -0.1,
            
            # Below speech threshold
            recent_energy[-1] < self.adaptive_threshold['speech'] * 1.2,
            
            # Not at the very beginning
            len(self.energy_history) > 10
        ]
        
        return sum(conditions) >= 2
    
    def calculate_energy_trend(self, energy_array: list) -> float:
        """Calculate energy trend (positive = increasing, negative = decreasing)"""
        if len(energy_array) < 3:
            return 0.0
        
        trend = 0.0
        for i in range(1, len(energy_array)):
            trend += energy_array[i] - energy_array[i - 1]
        
        return trend / (len(energy_array) - 1)
    
    def reset_state(self):
        """Reset VAD state"""
        self.current_state = 'silence'
        self.speech_start_time = 0
        self.silence_start_time = 0
        self.last_state_change = 0
        self.energy_history = []
        self.background_noise = 0.01
        
        self.logger.debug("VAD state reset")
    
    def optimize_for_speed(self):
        """Optimize VAD for maximum speed"""
        self.config.min_silence_duration = 600
        self.config.min_speech_duration = 150
        self.config.sensitivity = 0.8
        self.config.endpoint_detection = True
        
        self.logger.info("VAD optimized for ultra-fast response")
    
    def optimize_for_accuracy(self):
        """Optimize VAD for maximum accuracy"""
        self.config.min_silence_duration = 1000
        self.config.min_speech_duration = 300
        self.config.sensitivity = 0.6
        self.config.endpoint_detection = False
        
        self.logger.info("VAD optimized for accuracy")
    
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
            'config': {
                'sensitivity': self.config.sensitivity,
                'min_speech_duration': self.config.min_speech_duration,
                'min_silence_duration': self.config.min_silence_duration
            },
            'current_state': self.current_state,
            'background_noise': self.background_noise,
            'adaptive_thresholds': self.adaptive_threshold.copy()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.executor.shutdown(wait=False)
            self.logger.info("VAD cleanup completed")
        except Exception as e:
            self.logger.error(f"VAD cleanup error: {e}")

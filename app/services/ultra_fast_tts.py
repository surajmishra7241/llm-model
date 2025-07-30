# app/services/ultra_fast_tts.py - Ultra-fast Text-to-Speech with <400ms target
import asyncio
import time
import logging
import base64
import io
import os
from typing import Dict, Any, Optional
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
import threading

class UltraFastTTS:
    """Ultra-fast TTS with <400ms target processing time"""
    
    def __init__(self, model: str = 'neural-voice-fast', sample_rate: int = 24000, target_time: float = 0.4):
        self.model = model
        self.sample_rate = sample_rate
        self.target_time = target_time
        self.logger = logging.getLogger(__name__)
        
        # OpenAI TTS configuration (fastest option)
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_url = "https://api.openai.com/v1/audio/speech"
        
        # Performance settings
        self.executor = ThreadPoolExecutor(max_workers=6)
        self.session = requests.Session()
        
        # Voice configurations
        self.voice_models = {
            'alloy': {'speed': 'fast', 'personality': 'neutral'},
            'echo': {'speed': 'fast', 'personality': 'friendly'},
            'fable': {'speed': 'medium', 'personality': 'warm'},
            'onyx': {'speed': 'fast', 'personality': 'professional'},
            'nova': {'speed': 'fast', 'personality': 'energetic'},
            'shimmer': {'speed': 'fast', 'personality': 'calm'}
        }
        
        # Performance tracking
        self.processing_times = []
        self.success_rate = 100.0
        self.cache = {}  # Audio cache
    
    async def initialize_tts(self):
        """Initialize TTS system"""
        try:
            # Test TTS endpoint
            await self.test_tts_connection()
            
            # Warm up with short text
            await self.warmup_tts()
            
            self.logger.info("TTS system initialized")
            
        except Exception as e:
            self.logger.error(f"TTS initialization failed: {e}")
            raise
    
    async def test_tts_connection(self):
        """Test TTS connection"""
        loop = asyncio.get_event_loop()
        
        def _test():
            if not self.openai_api_key:
                raise Exception("OpenAI API key not found")
            
            # Test with minimal request
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'tts-1',
                'input': 'test',
                'voice': 'alloy',
                'response_format': 'opus',
                'speed': 1.0
            }
            
            response = self.session.post(
                self.openai_url,
                headers=headers,
                json=payload,
                timeout=5.0
            )
            
            if response.status_code != 200:
                raise Exception(f"TTS test failed: {response.status_code}")
            
            return True
        
        await loop.run_in_executor(self.executor, _test)
    
    async def warmup_tts(self):
        """Warm up TTS with dummy request"""
        try:
            await self.synthesize_async("Hello", {'voice': 'alloy'})
            self.logger.info("TTS warmup completed")
        except Exception as e:
            self.logger.warning(f"TTS warmup failed: {e}")
    
    async def synthesize_async(self, text: str, voice_config: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize speech asynchronously with ultra-fast processing"""
        start_time = time.time()
        
        try:
            # Validate and prepare text
            clean_text = self.prepare_text(text)
            if not clean_text:
                raise ValueError("Empty or invalid text")
            
            # Check cache
            cache_key = self.generate_cache_key(clean_text, voice_config)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                self.logger.debug("Using cached TTS audio")
                return cached_result
            
            # Synthesize audio
            result = await self.call_tts_api(clean_text, voice_config)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Update success rate
            if processing_time <= self.target_time:
                self.success_rate = (self.success_rate * 0.9) + 10.0
            else:
                self.success_rate = (self.success_rate * 0.9) + 0.0
            
            # Cache result (limit cache size)
            if len(self.cache) < 50:
                self.cache[cache_key] = result
            
            result['processing_time'] = processing_time
            result['within_target'] = processing_time <= self.target_time
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"TTS failed in {processing_time:.3f}s: {e}")
            raise Exception(f"Text-to-speech failed: {str(e)}")
    
    def prepare_text(self, text: str) -> str:
        """Prepare text for TTS optimization"""
        if not text or not text.strip():
            return ""
        
        # Clean text
        clean_text = text.strip()
        
        # Limit length for speed (keep under 500 characters)
        if len(clean_text) > 500:
            clean_text = clean_text[:497] + "..."
        
        # Remove problematic characters
        clean_text = clean_text.replace('\n', ' ').replace('\r', ' ')
        clean_text = ' '.join(clean_text.split())  # Normalize whitespace
        
        return clean_text
    
    async def call_tts_api(self, text: str, voice_config: Dict[str, Any]) -> Dict[str, Any]:
        """Call TTS API with optimization"""
        loop = asyncio.get_event_loop()
        
        def _call_tts():
            try:
                # Prepare voice settings
                voice = voice_config.get('voice', 'alloy')
                if voice not in self.voice_models:
                    voice = 'alloy'
                
                speaking_rate = voice_config.get('speaking_rate', 1.0)
                # Clamp speaking rate for quality
                speaking_rate = max(0.7, min(1.5, speaking_rate))
                
                # Use fastest model for speed
                model = 'tts-1'  # Faster than tts-1-hd
                
                headers = {
                    'Authorization': f'Bearer {self.openai_api_key}',
                    'Content-Type': 'application/json'
                }
                
                payload = {
                    'model': model,
                    'input': text,
                    'voice': voice,
                    'response_format': 'opus',  # Most efficient format
                    'speed': speaking_rate
                }
                
                # Make request with timeout
                response = self.session.post(
                    self.openai_url,
                    headers=headers,
                    json=payload,
                    timeout=3.0  # 3 second timeout
                )
                
                if response.status_code != 200:
                    raise Exception(f"TTS API error: {response.status_code} - {response.text}")
                
                # Process audio data
                audio_data = response.content
                
                # Convert to base64 for transmission
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                return {
                    'audio_data': audio_base64,
                    'format': 'opus',
                    'sample_rate': 24000,
                    'channels': 1,
                    'voice': voice,
                    'model': model
                }
                
            except Exception as e:
                self.logger.error(f"TTS API call failed: {e}")
                raise
        
        return await loop.run_in_executor(self.executor, _call_tts)
    
    def generate_cache_key(self, text: str, voice_config: Dict[str, Any]) -> str:
        """Generate cache key for TTS request"""
        import hashlib
        
        # Create hash from text and voice config
        content = json.dumps({
            'text': text,
            'voice': voice_config.get('voice', 'alloy'),
            'speaking_rate': voice_config.get('speaking_rate', 1.0)
        }, sort_keys=True)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.processing_times:
            return {
                'average_time': 0.0,
                'success_rate': 100.0,
                'target_achievement': 100.0,
                'total_requests': 0,
                'cache_size': len(self.cache)
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
            'max_time': max(self.processing_times),
            'cache_size': len(self.cache)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.session.close()
            self.executor.shutdown(wait=False)
            self.cache.clear()
            self.logger.info("TTS cleanup completed")
        except Exception as e:
            self.logger.error(f"TTS cleanup error: {e}")

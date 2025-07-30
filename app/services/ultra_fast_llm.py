# app/services/ultra_fast_llm.py - Ultra-fast LLM processing with <500ms target
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
import requests
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model_name: str = 'deepseek-r1:1.5b'
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1
    timeout: float = 5.0

class UltraFastLLM:
    """Ultra-fast LLM processing with <500ms target response time"""
    
    def __init__(self, model_name: str = 'deepseek-r1:1.5b', max_tokens: int = 100, target_time: float = 0.5):
        self.config = LLMConfig(
            model_name=model_name,
            max_tokens=max_tokens
        )
        self.target_time = target_time
        self.logger = logging.getLogger(__name__)
        
        # Ollama configuration
        self.ollama_url = "http://localhost:11434"
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Performance tracking
        self.processing_times = []
        self.success_rate = 100.0
        self.cache = {}  # Simple response cache
        
        # Connection pooling
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
    
    async def initialize_model(self):
        """Initialize and warm up the LLM model"""
        try:
            # Pull model if not exists
            await self.ensure_model_available()
            
            # Warm up model
            await self.warmup_model()
            
            self.logger.info(f"LLM model {self.config.model_name} initialized")
            
        except Exception as e:
            self.logger.error(f"LLM initialization failed: {e}")
            raise
    
    async def ensure_model_available(self):
        """Ensure model is available in Ollama"""
        loop = asyncio.get_event_loop()
        
        def _check_model():
            try:
                response = self.session.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [model['name'] for model in models]
                    
                    if self.config.model_name not in model_names:
                        self.logger.info(f"Pulling model {self.config.model_name}")
                        pull_response = self.session.post(
                            f"{self.ollama_url}/api/pull",
                            json={'name': self.config.model_name}
                        )
                        if pull_response.status_code != 200:
                            raise Exception(f"Failed to pull model: {pull_response.text}")
                    
                    return True
                else:
                    raise Exception(f"Ollama not available: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"Model availability check failed: {e}")
                raise
        
        await loop.run_in_executor(self.executor, _check_model)
    
    async def warmup_model(self):
        """Warm up model with dummy request"""
        try:
            dummy_messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'Hello'}
            ]
            
            await self.generate_async(
                messages=dummy_messages,
                max_tokens=10,
                temperature=0.0
            )
            
            self.logger.info("LLM warmup completed")
            
        except Exception as e:
            self.logger.warning(f"LLM warmup failed: {e}")
    
    async def generate_async(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None, stream: bool = False) -> Dict[str, Any]:
        """Generate response asynchronously with ultra-fast processing"""
        start_time = time.time()
        
        try:
            # Use cache key for repeated requests
            cache_key = self.generate_cache_key(messages, max_tokens, temperature)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                self.logger.debug("Using cached LLM response")
                return cached_result
            
            # Generate response
            result = await self.call_ollama_async(messages, max_tokens, temperature, stream)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Update success rate
            if processing_time <= self.target_time:
                self.success_rate = (self.success_rate * 0.9) + 10.0
            else:
                self.success_rate = (self.success_rate * 0.9) + 0.0
            
            # Cache successful results
            if result and len(self.cache) < 100:  # Limit cache size
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"LLM failed in {processing_time:.3f}s: {e}")
            raise Exception(f"LLM generation failed: {str(e)}")
    
    async def call_ollama_async(self, messages: List[Dict[str, str]], max_tokens: Optional[int],
                              temperature: Optional[float], stream: bool) -> Dict[str, Any]:
        """Call Ollama API asynchronously"""
        loop = asyncio.get_event_loop()
        
        def _call_ollama():
            try:
                # Prepare request payload
                payload = {
                    'model': self.config.model_name,
                    'messages': messages,
                    'stream': stream,
                    'options': {
                        'num_predict': max_tokens or self.config.max_tokens,
                        'temperature': temperature or self.config.temperature,
                        'top_p': self.config.top_p,
                        'frequency_penalty': self.config.frequency_penalty,
                        'presence_penalty': self.config.presence_penalty,
                        
                        # Ultra-fast optimizations
                        'num_ctx': 2048,  # Reduced context window
                        'num_batch': 512,
                        'num_gpu_layers': -1,  # Use all GPU layers
                        'use_mmap': True,
                        'use_mlock': False,
                    }
                }
                
                # Make request
                response = self.session.post(
                    f"{self.ollama_url}/api/chat",
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
                result = response.json()
                
                return {
                    'content': result['message']['content'],
                    'tokens_used': result.get('eval_count', 0),
                    'model': self.config.model_name,
                    'finish_reason': 'completed'
                }
                
            except Exception as e:
                self.logger.error(f"Ollama API call failed: {e}")
                raise
        
        return await loop.run_in_executor(self.executor, _call_ollama)
    
    def generate_cache_key(self, messages: List[Dict[str, str]], max_tokens: Optional[int],
                          temperature: Optional[float]) -> str:
        """Generate cache key for request"""
        import hashlib
        
        # Create hash from messages and parameters
        content = json.dumps({
            'messages': messages[-2:],  # Only last 2 messages for cache
            'max_tokens': max_tokens or self.config.max_tokens,
            'temperature': temperature or self.config.temperature,
            'model': self.config.model_name
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
            self.logger.info("LLM cleanup completed")
        except Exception as e:
            self.logger.error(f"LLM cleanup error: {e}")

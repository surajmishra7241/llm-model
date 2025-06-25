# app/services/llm_service.py
import requests
import logging
from typing import Optional, Dict, Any, List
from app.config import settings
import asyncio
import aiohttp
import hashlib
import numpy as np

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(self):
        self.base_url = str(settings.OLLAMA_URL).rstrip('/')
        self.default_model = settings.DEFAULT_OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT
        # Use the main model for embeddings if no specific embedding model
        self.embedding_model = getattr(settings, 'EMBEDDING_MODEL', 'deepseek-r1:1.5b')

    async def _make_request_async(self, endpoint: str, payload: dict) -> dict:
        """Make async HTTP request to Ollama"""
        try:
            url = f"{self.base_url}{endpoint}"
            logger.debug(f"Making request to: {url}")
            logger.debug(f"Payload: {payload}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_text = await response.text()
                    logger.debug(f"Response status: {response.status}")
                    logger.debug(f"Response: {response_text[:500]}...")
                    
                    if response.status == 405:
                        logger.error(f"Method not allowed for {url}. Available methods might be different.")
                        raise requests.exceptions.HTTPError(f"405 Client Error: Method Not Allowed for url: {url}")
                    
                    response.raise_for_status()
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP Client Error calling Ollama: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            raise

    async def _make_request(self, endpoint: str, payload: dict) -> dict:
        """Fallback sync request method"""
        try:
            url = f"{self.base_url}{endpoint}"
            logger.debug(f"Making sync request to: {url}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response: {response.text[:500]}...")
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            raise

    async def check_ollama_status(self) -> Dict[str, Any]:
        """Check if Ollama is running and accessible"""
        try:
            # Try to get the list of models first
            models = await self.list_models()
            
            # Check if our models are available
            model_names = [model.get('name', '') for model in models]
            
            return {
                "status": "healthy",
                "available_models": model_names,
                "default_model_available": self.default_model in model_names,
                "embedding_model_available": self.embedding_model in model_names,
                "base_url": self.base_url
            }
            
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "base_url": self.base_url
            }

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[list] = None
    ) -> Dict[str, Any]:
        payload = {
            "model": model or self.default_model,
            "prompt": prompt,
            "stream": False,
            "options": options or {},
        }
        if system: 
            payload["system"] = system
        if template: 
            payload["template"] = template
        if context: 
            payload["context"] = context
        
        try:
            return await self._make_request_async("/api/generate", payload)
        except Exception as e:
            logger.warning(f"Async request failed, trying sync: {str(e)}")
            return await self._make_request("/api/generate", payload)

    async def chat(
        self,
        messages: list,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "options": options or {},
        }
        
        try:
            return await self._make_request_async("/api/chat", payload)
        except Exception as e:
            logger.warning(f"Async request failed, trying sync: {str(e)}")
            return await self._make_request("/api/chat", payload)

    async def list_models(self) -> list:
        """List available models"""
        try:
            # Try async first
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}/api/tags",
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        return result.get("models", [])
            except:
                # Fallback to sync
                response = requests.get(
                    f"{self.base_url}/api/tags",
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json().get("models", [])
                
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []

    async def create_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """Create embeddings with multiple fallback strategies"""
        target_model = model or self.embedding_model
        
        # Strategy 1: Try the embeddings endpoint
        try:
            return await self._try_embeddings_endpoint(text, target_model)
        except Exception as e:
            logger.warning(f"Embeddings endpoint failed: {str(e)}")
        
        # Strategy 2: Try using the main model for embeddings if it's different
        if target_model != self.default_model:
            try:
                logger.info(f"Trying main model {self.default_model} for embeddings")
                return await self._try_embeddings_endpoint(text, self.default_model)
            except Exception as e:
                logger.warning(f"Main model embedding failed: {str(e)}")
        
        # Strategy 3: Try generate endpoint with special prompt
        try:
            return await self._create_embedding_via_generate(text, target_model)
        except Exception as e:
            logger.warning(f"Generate-based embedding failed: {str(e)}")
        
        # Strategy 4: Fallback to deterministic hash-based embedding
        logger.warning("All embedding methods failed, using deterministic fallback")
        return self._create_deterministic_embedding(text)

    async def _try_embeddings_endpoint(self, text: str, model: str) -> List[float]:
        """Try the official embeddings endpoint"""
        payload = {
            "model": model,
            "prompt": text  # Some versions use 'prompt' instead of 'input'
        }
        
        try:
            # Try with 'prompt' first
            result = await self._make_request_async("/api/embeddings", payload)
        except:
            # Try with 'input' 
            payload["input"] = payload.pop("prompt")
            try:
                result = await self._make_request_async("/api/embeddings", payload)
            except:
                # Try the /api/embed endpoint
                result = await self._make_request_async("/api/embed", payload)
        
        # Handle different response formats
        if 'embeddings' in result:
            embeddings = result['embeddings']
            if embeddings and len(embeddings) > 0:
                return embeddings[0] if isinstance(embeddings[0], list) else embeddings
        elif 'embedding' in result:
            return result['embedding']
        elif 'data' in result:
            # OpenAI-style response
            return result['data'][0]['embedding']
        
        raise ValueError("No embeddings found in response")

    async def _create_embedding_via_generate(self, text: str, model: str) -> List[float]:
        """Create embedding using generate endpoint with special prompt"""
        
        # Try a simple approach that might work with some models
        prompt = f"Embed: {text}"
        
        response = await self.generate(
            prompt=prompt,
            model=model,
            options={
                "temperature": 0.0,
                "num_predict": 1,  # Minimal generation
                "stop": ["\n", ".", "!"]
            }
        )
        
        # This is a fallback - extract features from the response
        response_text = response.get("response", "")
        
        # Create a more sophisticated embedding from the response
        return self._text_to_embedding(text + " " + response_text)

    def _create_deterministic_embedding(self, text: str) -> List[float]:
        """Create a deterministic embedding from text using multiple hash functions"""
        
        # Use multiple hash functions for better distribution
        hash_functions = [
            lambda x: hashlib.md5(x.encode()).hexdigest(),
            lambda x: hashlib.sha1(x.encode()).hexdigest(),
            lambda x: hashlib.sha256(x.encode()).hexdigest(),
        ]
        
        # Target dimension
        target_dim = getattr(settings, 'EMBEDDING_FALLBACK_DIMENSION', 768)
        embedding = []
        
        # Generate embedding using multiple hash functions
        for i, hash_func in enumerate(hash_functions):
            hash_hex = hash_func(f"{text}_{i}")
            
            # Convert hex to floats
            for j in range(0, len(hash_hex), 2):
                if len(embedding) >= target_dim:
                    break
                val = int(hash_hex[j:j+2], 16) / 255.0  # Normalize to 0-1
                embedding.append(val)
            
            if len(embedding) >= target_dim:
                break
        
        # Pad or truncate to desired dimension
        if len(embedding) < target_dim:
            embedding.extend([0.1] * (target_dim - len(embedding)))
        else:
            embedding = embedding[:target_dim]
        
        # Add some text-based features
        text_features = [
            len(text) / 1000.0,  # Text length feature
            text.count(' ') / len(text) if text else 0,  # Word density
            text.count('.') / len(text) if text else 0,  # Sentence density
        ]
        
        # Replace some values with text features
        for i, feature in enumerate(text_features):
            if i < len(embedding):
                embedding[i] = feature
        
        # Normalize the embedding
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding = (embedding_array / norm).tolist()
        
        logger.info(f"Created deterministic embedding with dimension {len(embedding)}")
        return embedding

    def _text_to_embedding(self, text: str) -> List[float]:
        """Convert text to embedding using various text features"""
        
        target_dim = getattr(settings, 'EMBEDDING_FALLBACK_DIMENSION', 768)
        
        # Extract various text features
        features = []
        
        # Character-level features
        for i in range(min(len(text), 50)):
            features.append(ord(text[i]) / 255.0)
        
        # Word-level features
        words = text.lower().split()
        for word in words[:20]:  # First 20 words
            word_hash = hash(word) % 1000
            features.append(word_hash / 1000.0)
        
        # Statistical features
        features.extend([
            len(text) / 1000.0,
            len(words) / 100.0 if words else 0,
            text.count(' ') / len(text) if text else 0,
            text.count('.') / len(text) if text else 0,
            text.count(',') / len(text) if text else 0,
        ])
        
        # Pad or truncate
        if len(features) < target_dim:
            features.extend([0.1] * (target_dim - len(features)))
        else:
            features = features[:target_dim]
        
        # Normalize
        features_array = np.array(features)
        norm = np.linalg.norm(features_array)
        if norm > 0:
            features = (features_array / norm).tolist()
        
        return features

    async def check_model_availability(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        try:
            models = await self.list_models()
            available_models = [model.get('name', '') for model in models]
            is_available = model_name in available_models
            logger.info(f"Model {model_name} availability: {is_available}")
            logger.info(f"Available models: {available_models}")
            return is_available
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            return False

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model if it's not available"""
        try:
            payload = {"name": model_name}
            
            # Use longer timeout for model pulling
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/pull",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes
                    ) as response:
                        response.raise_for_status()
                        logger.info(f"Successfully pulled model: {model_name}")
                        return True
            except:
                # Fallback to sync
                response = requests.post(
                    f"{self.base_url}/api/pull",
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()
                logger.info(f"Successfully pulled model: {model_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            return False

    async def ensure_embedding_model(self) -> bool:
        """Ensure embedding model is available"""
        try:
            # Check if current embedding model is available
            if await self.check_model_availability(self.embedding_model):
                logger.info(f"Embedding model {self.embedding_model} is available")
                return True
            
            # If not available and it's different from default model, try default model
            if self.embedding_model != self.default_model:
                if await self.check_model_availability(self.default_model):
                    logger.info(f"Using default model {self.default_model} for embeddings")
                    self.embedding_model = self.default_model
                    return True
            
            # Try to pull the embedding model
            logger.info(f"Embedding model {self.embedding_model} not found, attempting to pull...")
            success = await self.pull_model(self.embedding_model)
            
            if not success and self.embedding_model != self.default_model:
                logger.info(f"Failed to pull {self.embedding_model}, falling back to {self.default_model}")
                self.embedding_model = self.default_model
                return await self.check_model_availability(self.default_model)
            
            return success
            
        except Exception as e:
            logger.error(f"Error ensuring embedding model: {str(e)}")
            return False
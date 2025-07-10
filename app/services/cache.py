from redis.asyncio import Redis
from app.config import settings
import logging
from typing import Any, Optional
import json

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.redis = None
        self.enabled = settings.CACHE_ENABLED
        self._connection_attempted = False
        
    async def connect(self):
        if self.enabled and not self._connection_attempted:
            self._connection_attempted = True
            try:
                self.redis = Redis.from_url(
                    str(settings.REDIS_URL),
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                await self.redis.ping()
                logger.info("Connected to Redis successfully")
                self.enabled = True
            except Exception as e:
                logger.warning(f"Redis connection failed, caching disabled: {str(e)}")
                self.enabled = False
                self.redis = None
    
    async def disconnect(self):
        if self.redis:
            try:
                await self.redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}")
    
    async def get(self, key: str) -> Optional[Any]:
        if not self.enabled or not self.redis:
            return None
            
        try:
            data = await self.redis.get(key)
            if data is None:
                return None
                
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        except Exception as e:
            logger.warning(f"Cache get error for key '{key}': {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self.enabled or not self.redis:
            return False
            
        try:
            ttl = ttl or getattr(settings, 'CACHE_TTL', 300)  # Default 5 minutes
            
            if isinstance(value, (str, bytes)):
                serialized_value = value
            else:
                try:
                    serialized_value = json.dumps(value, default=str)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Failed to serialize value for key '{key}': {str(e)}")
                    return False
            
            await self.redis.set(key, serialized_value, ex=ttl)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key '{key}': {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        if not self.enabled or not self.redis:
            return False
            
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Cache delete error for key '{key}': {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        if not self.enabled or not self.redis:
            return False
            
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Cache exists error for key '{key}': {str(e)}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern"""
        if not self.enabled or not self.redis:
            return 0
            
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache clear pattern error for pattern '{pattern}': {str(e)}")
            return 0
    
    async def health_check(self) -> bool:
        """Check if Redis is healthy"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {str(e)}")
            return False

    async def get_chat_history(self, user_id: str, agent_id: str) -> Optional[list]:
        """Get chat history from cache"""
        if not self.enabled or not self.redis:
            return None
            
        try:
            data = await self.redis.get(f"chat_history:{user_id}:{agent_id}")
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.warning(f"Error getting chat history: {str(e)}")
            return None

    async def save_chat_history(self, user_id: str, agent_id: str, history: list) -> bool:
        """Save chat history to cache"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            await self.redis.set(
                f"chat_history:{user_id}:{agent_id}",
                json.dumps(history),
                ex=86400  # 24 hours TTL
            )
            return True
        except Exception as e:
            logger.warning(f"Error saving chat history: {str(e)}")
            return False

    async def clear_chat_history(self, user_id: str, agent_id: str) -> bool:
        """Clear chat history from cache"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            await self.redis.delete(f"chat_history:{user_id}:{agent_id}")
            return True
        except Exception as e:
            logger.warning(f"Error clearing chat history: {str(e)}")
            return False


# Global instance
cache_service = CacheService()
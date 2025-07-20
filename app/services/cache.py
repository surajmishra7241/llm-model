import json
import logging
from typing import Any, Optional, Dict, List
from fastapi.encoders import jsonable_encoder
from redis.asyncio import Redis
from app.config import settings
import asyncio
import time

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.enabled = settings.CACHE_ENABLED
        self._connection_attempted = False
        self._connection_lock = asyncio.Lock()
        
    async def connect(self):
        """Initialize Redis connection with proper error handling"""
        if not self.enabled or self._connection_attempted:
            return
            
        async with self._connection_lock:
            if self._connection_attempted:
                return
                
            self._connection_attempted = True
            
            try:
                self.redis = Redis.from_url(
                    str(settings.REDIS_URL),
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                    max_connections=20
                )
                
                # Test connection
                await self.redis.ping()
                logger.info("✅ Connected to Redis successfully")
                self.enabled = True
                
            except Exception as e:
                logger.warning(f"❌ Redis connection failed, caching disabled: {str(e)}")
                self.enabled = False
                self.redis = None
    
    async def disconnect(self):
        """Clean disconnect from Redis"""
        if self.redis:
            try:
                await self.redis.close()
                logger.info("Redis connection closed gracefully")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}")
            finally:
                self.redis = None
                self._connection_attempted = False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with automatic JSON decoding"""
        if not self.enabled or not self.redis:
            return None
            
        try:
            data = await self.redis.get(key)
            if data is None:
                return None
                
            # Try to parse as JSON first
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                # Return as string if not valid JSON
                return data
                
        except Exception as e:
            logger.warning(f"Cache get error for key '{key}': {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with automatic JSON encoding"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            ttl = ttl or getattr(settings, 'CACHE_TTL', 300)
            
            # Handle different value types
            if isinstance(value, (str, bytes)):
                serialized_value = value
            else:
                try:
                    # Use jsonable_encoder for complex objects
                    serializable_value = jsonable_encoder(value)
                    serialized_value = json.dumps(serializable_value, default=str, ensure_ascii=False)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Failed to serialize value for key '{key}': {str(e)}")
                    return False
            
            await self.redis.set(key, serialized_value, ex=ttl)
            return True
            
        except Exception as e:
            logger.warning(f"Cache set error for key '{key}': {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Cache delete error for key '{key}': {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            logger.warning(f"Cache exists error for key '{key}': {str(e)}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for existing key"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            result = await self.redis.expire(key, ttl)
            return result
        except Exception as e:
            logger.warning(f"Cache expire error for key '{key}': {str(e)}")
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
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for a key"""
        if not self.enabled or not self.redis:
            return None
            
        try:
            ttl = await self.redis.ttl(key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.warning(f"Cache TTL error for key '{key}': {str(e)}")
            return None
    
    # JSON-specific methods with better error handling
    async def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set JSON data with proper serialization"""
        if not self.enabled or not self.redis:
            return False
            
        try:
            # Use FastAPI's jsonable_encoder for consistent serialization
            serializable_value = jsonable_encoder(value)
            json_string = json.dumps(serializable_value, default=str, ensure_ascii=False)
            
            ttl = ttl or getattr(settings, 'CACHE_TTL', 300)
            await self.redis.set(key, json_string, ex=ttl)
            return True
        except (TypeError, ValueError) as e:
            logger.warning(f"JSON serialization failed for key '{key}': {str(e)}")
            return False
        except Exception as e:
            logger.warning(f"Cache set JSON error for key '{key}': {str(e)}")
            return False
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON data with proper deserialization"""
        if not self.enabled or not self.redis:
            return None
            
        try:
            data = await self.redis.get(key)
            if data is None:
                return None
            return json.loads(data)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON deserialization failed for key '{key}': {str(e)}")
            return None
        except Exception as e:
            logger.warning(f"Cache get JSON error for key '{key}': {str(e)}")
            return None
    
    # Chat history specific methods
    async def get_chat_history(self, user_id: str, agent_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get conversation history for user and agent"""
        key = f"conversation:{user_id}:{agent_id}"
        return await self.get_json(key) or []
    
    async def save_chat_history(
        self, 
        user_id: str, 
        agent_id: str, 
        history: List[Dict[str, Any]], 
        ttl: Optional[int] = None
    ) -> bool:
        """Save conversation history to cache"""
        key = f"conversation:{user_id}:{agent_id}"
        ttl = ttl or 86400  # 24 hours default
        return await self.set_json(key, history, ttl)
    
    async def clear_chat_history(self, user_id: str, agent_id: str) -> bool:
        """Clear conversation history for user and agent"""
        key = f"conversation:{user_id}:{agent_id}"
        return await self.delete(key)
    
    async def append_to_chat_history(
        self,
        user_id: str,
        agent_id: str,
        new_messages: List[Dict[str, Any]],
        max_history: int = 20
    ) -> bool:
        """Append messages to chat history with size limiting"""
        try:
            # Get existing history
            history = await self.get_chat_history(user_id, agent_id)
            
            # Append new messages
            history.extend(new_messages)
            
            # Keep only recent messages
            if len(history) > max_history:
                history = history[-max_history:]
            
            # Save updated history
            return await self.save_chat_history(user_id, agent_id, history)
            
        except Exception as e:
            logger.error(f"Error appending to chat history: {str(e)}")
            return False
    
    # Agent-specific caching
    async def cache_agent_response(
        self,
        agent_id: str,
        user_id: str,
        query_hash: str,
        response: str,
        ttl: int = 1800  # 30 minutes
    ) -> bool:
        """Cache agent response for similar queries"""
        key = f"agent_response:{agent_id}:{user_id}:{query_hash}"
        return await self.set(key, response, ttl)
    
    async def get_cached_agent_response(
        self,
        agent_id: str,
        user_id: str,
        query_hash: str
    ) -> Optional[str]:
        """Get cached agent response"""
        key = f"agent_response:{agent_id}:{user_id}:{query_hash}"
        return await self.get(key)
    
    # System-wide cache operations
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache service"""
        if not self.enabled or not self.redis:
            return {
                "status": "disabled",
                "enabled": False,
                "redis_connected": False
            }
            
        try:
            # Test basic operations
            start_time = time.time()
            await self.redis.ping()
            ping_time = (time.time() - start_time) * 1000
            
            # Get some stats
            info = await self.redis.info()
            
            return {
                "status": "healthy",
                "enabled": True,
                "redis_connected": True,
                "ping_time_ms": round(ping_time, 2),
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "enabled": self.enabled,
                "redis_connected": False,
                "error": str(e)
            }
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled or not self.redis:
            return {"cache_enabled": False}
            
        try:
            info = await self.redis.info()
            return {
                "cache_enabled": True,
                "total_connections_received": info.get("total_connections_received", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) / 
                    max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1)
                ) * 100,
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B")
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"cache_enabled": True, "error": str(e)}

# Global cache service instance
cache_service = CacheService()

# Utility functions for common cache patterns
async def cache_with_key(key: str, fetch_func, ttl: int = 300):
    """Generic cache-aside pattern implementation"""
    # Try to get from cache first
    cached_value = await cache_service.get(key)
    if cached_value is not None:
        return cached_value
    
    # Fetch from source
    try:
        value = await fetch_func() if asyncio.iscoroutinefunction(fetch_func) else fetch_func()
        # Cache the result
        await cache_service.set(key, value, ttl)
        return value
    except Exception as e:
        logger.error(f"Error in cache_with_key for '{key}': {str(e)}")
        raise

def generate_cache_key(*parts: str) -> str:
    """Generate a standardized cache key from parts"""
    return ":".join(str(part) for part in parts if part)

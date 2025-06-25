from typing import Any, Optional, Callable, Awaitable
from functools import wraps
from app.services.cache import cache_service
import json
import hashlib
import inspect

def cached(key_pattern: str, ttl: Optional[int] = None):
    """Decorator for caching async function results with smart key generation"""
    def decorator(func: Callable[..., Awaitable[Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not cache_service.enabled:
                return await func(*args, **kwargs)
            
            try:
                # Get function signature to map args to parameter names
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Create a dictionary of all parameters
                params = dict(bound_args.arguments)
                
                # Generate query_hash if needed and not provided
                if 'query_hash' in key_pattern and 'query_hash' not in params:
                    query = params.get('query', '')
                    if query:
                        params['query_hash'] = hashlib.md5(query.encode()).hexdigest()[:8]
                    else:
                        params['query_hash'] = 'no_query'
                
                # Format the cache key with available parameters
                try:
                    cache_key = key_pattern.format(**params)
                except KeyError as e:
                    # If formatting fails, create a simple fallback key
                    func_name = func.__name__
                    args_hash = hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()[:8]
                    cache_key = f"{func_name}:{args_hash}"
                
                # Try to get from cache
                cached_data = await cache_service.get(cache_key)
                if cached_data is not None:
                    return cached_data
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await cache_service.set(cache_key, result, ttl)
                return result
                
            except Exception as e:
                # If caching fails, just execute the function
                return await func(*args, **kwargs)
                
        return wrapper
    return decorator
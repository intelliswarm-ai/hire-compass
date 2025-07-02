"""
Cache adapter implementations.

Provides both Redis and in-memory cache implementations following the Cache protocol.
Includes features like TTL support, key namespacing, and automatic serialization.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Set, Tuple, Union

import redis.asyncio as redis
from redis.asyncio import ConnectionPool
from redis.exceptions import RedisError

from src.shared.protocols import Cache, Logger
from src.shared.types import Result, Success, Failure


class CacheKeyBuilder:
    """Utility for building consistent cache keys."""
    
    def __init__(self, namespace: str = "hr_matcher"):
        self.namespace = namespace
    
    def build(self, *parts: Union[str, int, float]) -> str:
        """Build cache key from parts."""
        clean_parts = [str(p).replace(":", "_") for p in parts]
        return f"{self.namespace}:{':'.join(clean_parts)}"
    
    def parse(self, key: str) -> Tuple[str, ...]:
        """Parse cache key into parts."""
        parts = key.split(":")
        if parts[0] == self.namespace:
            return tuple(parts[1:])
        return tuple(parts)


class RedisCache(Cache):
    """Redis-based cache implementation."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: Optional[int] = 3600,
        key_prefix: str = "hr_matcher",
        connection_pool_kwargs: Optional[Dict[str, Any]] = None,
        logger: Optional[Logger] = None,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.key_builder = CacheKeyBuilder(key_prefix)
        self.logger = logger
        
        # Connection pool configuration
        pool_kwargs = connection_pool_kwargs or {}
        pool_kwargs.update({
            "host": host,
            "port": port,
            "db": db,
            "password": password,
            "decode_responses": False,  # We'll handle encoding/decoding
            "max_connections": pool_kwargs.get("max_connections", 50),
            "socket_connect_timeout": pool_kwargs.get("socket_connect_timeout", 5),
            "socket_timeout": pool_kwargs.get("socket_timeout", 5),
            "retry_on_timeout": pool_kwargs.get("retry_on_timeout", True),
        })
        
        self._pool = ConnectionPool(**pool_kwargs)
        self._client: Optional[redis.Redis] = None
    
    async def _get_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._client is None:
            self._client = redis.Redis(connection_pool=self._pool)
        return self._client
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Try JSON first for better interoperability
            return json.dumps(value).encode("utf-8")
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        full_key = self.key_builder.build(key)
        
        try:
            client = await self._get_client()
            data = await client.get(full_key)
            
            if data is None:
                if self.logger:
                    self.logger.debug(f"Cache miss for key: {key}")
                return None
            
            value = self._deserialize(data)
            if self.logger:
                self.logger.debug(f"Cache hit for key: {key}")
            return value
            
        except RedisError as e:
            if self.logger:
                self.logger.error(f"Redis error on get: {key}", error=e)
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error on cache get: {key}", error=e)
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL in seconds."""
        full_key = self.key_builder.build(key)
        ttl = ttl or self.default_ttl
        
        try:
            client = await self._get_client()
            data = self._serialize(value)
            
            if ttl:
                await client.setex(full_key, ttl, data)
            else:
                await client.set(full_key, data)
            
            if self.logger:
                self.logger.debug(f"Cache set for key: {key}, ttl: {ttl}")
                
        except RedisError as e:
            if self.logger:
                self.logger.error(f"Redis error on set: {key}", error=e)
            raise
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error on cache set: {key}", error=e)
            raise
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        full_key = self.key_builder.build(key)
        
        try:
            client = await self._get_client()
            await client.delete(full_key)
            
            if self.logger:
                self.logger.debug(f"Cache delete for key: {key}")
                
        except RedisError as e:
            if self.logger:
                self.logger.error(f"Redis error on delete: {key}", error=e)
            raise
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        full_key = self.key_builder.build(key)
        
        try:
            client = await self._get_client()
            return bool(await client.exists(full_key))
            
        except RedisError as e:
            if self.logger:
                self.logger.error(f"Redis error on exists: {key}", error=e)
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries with our prefix."""
        try:
            client = await self._get_client()
            pattern = self.key_builder.build("*")
            
            # Use SCAN for better performance with large datasets
            cursor = 0
            while True:
                cursor, keys = await client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                
                if keys:
                    await client.delete(*keys)
                
                if cursor == 0:
                    break
            
            if self.logger:
                self.logger.info("Cache cleared")
                
        except RedisError as e:
            if self.logger:
                self.logger.error("Redis error on clear", error=e)
            raise
    
    async def get_many(self, keys: list[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not keys:
            return {}
        
        full_keys = [self.key_builder.build(k) for k in keys]
        
        try:
            client = await self._get_client()
            values = await client.mget(full_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
            
            return result
            
        except RedisError as e:
            if self.logger:
                self.logger.error("Redis error on get_many", error=e)
            return {}
    
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Set multiple values in cache."""
        if not items:
            return
        
        ttl = ttl or self.default_ttl
        
        try:
            client = await self._get_client()
            
            # Use pipeline for atomic operations
            async with client.pipeline() as pipe:
                for key, value in items.items():
                    full_key = self.key_builder.build(key)
                    data = self._serialize(value)
                    
                    if ttl:
                        pipe.setex(full_key, ttl, data)
                    else:
                        pipe.set(full_key, data)
                
                await pipe.execute()
            
            if self.logger:
                self.logger.debug(f"Cache set_many for {len(items)} items")
                
        except RedisError as e:
            if self.logger:
                self.logger.error("Redis error on set_many", error=e)
            raise
    
    async def increment(self, key: str, delta: int = 1) -> int:
        """Increment numeric value."""
        full_key = self.key_builder.build(key)
        
        try:
            client = await self._get_client()
            return await client.incrby(full_key, delta)
            
        except RedisError as e:
            if self.logger:
                self.logger.error(f"Redis error on increment: {key}", error=e)
            raise
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key."""
        full_key = self.key_builder.build(key)
        
        try:
            client = await self._get_client()
            return await client.expire(full_key, ttl)
            
        except RedisError as e:
            if self.logger:
                self.logger.error(f"Redis error on expire: {key}", error=e)
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
        await self._pool.disconnect()


class InMemoryCache(Cache):
    """Thread-safe in-memory cache implementation with TTL support."""
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: Optional[int] = 3600,
        cleanup_interval: int = 300,
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        
        # Use OrderedDict for LRU eviction
        self._data: OrderedDict[str, Tuple[Any, Optional[float]]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup()
    
    def _start_cleanup(self) -> None:
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        async with self._lock:
            current_time = time.time()
            keys_to_delete = []
            
            for key, (_, expiry) in self._data.items():
                if expiry and expiry < current_time:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._data[key]
    
    async def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._data) >= self.max_size:
            # Remove oldest item (first in OrderedDict)
            self._data.popitem(last=False)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key not in self._data:
                return None
            
            value, expiry = self._data[key]
            
            # Check expiration
            if expiry and expiry < time.time():
                del self._data[key]
                return None
            
            # Move to end for LRU
            self._data.move_to_end(key)
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL in seconds."""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl if ttl else None
        
        async with self._lock:
            # Update existing or add new
            if key in self._data:
                self._data.move_to_end(key)
            else:
                await self._evict_if_needed()
            
            self._data[key] = (value, expiry)
    
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        async with self._lock:
            self._data.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        async with self._lock:
            if key not in self._data:
                return False
            
            _, expiry = self._data[key]
            if expiry and expiry < time.time():
                del self._data[key]
                return False
            
            return True
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._data.clear()
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class CacheDecorator:
    """Decorator for caching function results."""
    
    def __init__(
        self,
        cache: Cache,
        key_prefix: str,
        ttl: Optional[int] = None,
        key_builder: Optional[Callable[..., str]] = None,
    ):
        self.cache = cache
        self.key_prefix = key_prefix
        self.ttl = ttl
        self.key_builder = key_builder or self._default_key_builder
    
    def _default_key_builder(self, *args, **kwargs) -> str:
        """Build cache key from function arguments."""
        parts = [self.key_prefix]
        parts.extend(str(arg) for arg in args)
        parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return ":".join(parts)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorate async function with caching."""
        async def wrapper(*args, **kwargs):
            # Build cache key
            cache_key = self.key_builder(*args, **kwargs)
            
            # Try to get from cache
            cached_value = await self.cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            await self.cache.set(cache_key, result, self.ttl)
            
            return result
        
        return wrapper
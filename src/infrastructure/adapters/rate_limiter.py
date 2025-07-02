"""
Rate limiter implementation for API protection.

Provides token bucket and sliding window rate limiting strategies
to protect services from overload and ensure fair usage.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, Optional, Tuple

from src.shared.protocols import Cache, Logger, RateLimiter


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    key: str
    limit: int
    window: float
    retry_after: Optional[float] = None
    
    def __str__(self) -> str:
        msg = f"Rate limit exceeded for key '{self.key}': {self.limit} requests per {self.window}s"
        if self.retry_after:
            msg += f" (retry after {self.retry_after:.1f}s)"
        return msg


@dataclass
class RateLimitInfo:
    """Information about current rate limit status."""
    
    limit: int
    remaining: int
    reset_at: datetime
    retry_after: Optional[float] = None
    
    @property
    def is_exceeded(self) -> bool:
        """Check if rate limit is exceeded."""
        return self.remaining <= 0
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP rate limit headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at.timestamp())),
        }
        
        if self.retry_after:
            headers["Retry-After"] = str(int(self.retry_after))
        
        return headers


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    
    def __post_init__(self) -> None:
        """Initialize bucket with full capacity."""
        self.tokens = float(self.capacity)
        self.last_refill = time.time()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens to be available."""
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class TokenBucketRateLimiter(RateLimiter):
    """Token bucket rate limiter implementation."""
    
    def __init__(
        self,
        rate: int,  # requests per second
        burst: Optional[int] = None,  # burst capacity
        cache: Optional[Cache] = None,
        key_prefix: str = "rate_limit",
        logger: Optional[Logger] = None,
    ):
        self.rate = rate
        self.burst = burst or rate
        self.cache = cache
        self.key_prefix = key_prefix
        self.logger = logger
        
        # Local buckets for when cache is not available
        self._local_buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str, tokens: int = 1) -> bool:
        """Check if request is allowed."""
        try:
            return await self.consume(key, tokens)
        except RateLimitExceeded:
            return False
    
    async def consume(self, key: str, tokens: int = 1) -> bool:
        """Consume tokens from bucket."""
        if self.cache:
            return await self._consume_with_cache(key, tokens)
        else:
            return await self._consume_local(key, tokens)
    
    async def get_remaining(self, key: str) -> int:
        """Get remaining tokens."""
        if self.cache:
            bucket_data = await self._get_bucket_from_cache(key)
            if bucket_data:
                bucket = self._deserialize_bucket(bucket_data)
                bucket._refill()
                return int(bucket.tokens)
        else:
            async with self._lock:
                bucket = self._get_or_create_bucket(key)
                bucket._refill()
                return int(bucket.tokens)
        
        return self.burst
    
    async def get_info(self, key: str) -> RateLimitInfo:
        """Get rate limit info for key."""
        remaining = await self.get_remaining(key)
        
        # Calculate reset time (when bucket will be full)
        if remaining < self.burst:
            tokens_to_full = self.burst - remaining
            seconds_to_full = tokens_to_full / self.rate
            reset_at = datetime.now() + timedelta(seconds=seconds_to_full)
        else:
            reset_at = datetime.now()
        
        retry_after = None
        if remaining <= 0:
            retry_after = 1.0 / self.rate  # Time for 1 token
        
        return RateLimitInfo(
            limit=self.burst,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
        )
    
    async def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        if self.cache:
            cache_key = f"{self.key_prefix}:{key}"
            await self.cache.delete(cache_key)
        else:
            async with self._lock:
                self._local_buckets.pop(key, None)
        
        if self.logger:
            self.logger.info(f"Rate limit reset for key: {key}")
    
    def _get_or_create_bucket(self, key: str) -> TokenBucket:
        """Get or create local token bucket."""
        if key not in self._local_buckets:
            self._local_buckets[key] = TokenBucket(
                capacity=self.burst,
                refill_rate=self.rate,
            )
        return self._local_buckets[key]
    
    async def _consume_local(self, key: str, tokens: int) -> bool:
        """Consume tokens from local bucket."""
        async with self._lock:
            bucket = self._get_or_create_bucket(key)
            
            if bucket.consume(tokens):
                return True
            
            wait_time = bucket.get_wait_time(tokens)
            raise RateLimitExceeded(
                key=key,
                limit=self.burst,
                window=self.burst / self.rate,
                retry_after=wait_time,
            )
    
    async def _consume_with_cache(self, key: str, tokens: int) -> bool:
        """Consume tokens using cache for distributed rate limiting."""
        cache_key = f"{self.key_prefix}:{key}"
        
        # Try to get bucket from cache
        bucket_data = await self._get_bucket_from_cache(key)
        
        if bucket_data:
            bucket = self._deserialize_bucket(bucket_data)
        else:
            bucket = TokenBucket(
                capacity=self.burst,
                refill_rate=self.rate,
            )
        
        # Try to consume tokens
        if bucket.consume(tokens):
            # Save updated bucket to cache
            await self._save_bucket_to_cache(key, bucket)
            return True
        
        # Rate limit exceeded
        wait_time = bucket.get_wait_time(tokens)
        raise RateLimitExceeded(
            key=key,
            limit=self.burst,
            window=self.burst / self.rate,
            retry_after=wait_time,
        )
    
    async def _get_bucket_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get bucket data from cache."""
        cache_key = f"{self.key_prefix}:{key}"
        return await self.cache.get(cache_key)
    
    async def _save_bucket_to_cache(self, key: str, bucket: TokenBucket) -> None:
        """Save bucket data to cache."""
        cache_key = f"{self.key_prefix}:{key}"
        bucket_data = self._serialize_bucket(bucket)
        
        # Use TTL to automatically clean up unused buckets
        ttl = int(self.burst / self.rate * 2)  # 2x the time to fill bucket
        await self.cache.set(cache_key, bucket_data, ttl)
    
    def _serialize_bucket(self, bucket: TokenBucket) -> Dict[str, Any]:
        """Serialize bucket for storage."""
        return {
            "capacity": bucket.capacity,
            "refill_rate": bucket.refill_rate,
            "tokens": bucket.tokens,
            "last_refill": bucket.last_refill,
        }
    
    def _deserialize_bucket(self, data: Dict[str, Any]) -> TokenBucket:
        """Deserialize bucket from storage."""
        bucket = TokenBucket(
            capacity=data["capacity"],
            refill_rate=data["refill_rate"],
        )
        bucket.tokens = data["tokens"]
        bucket.last_refill = data["last_refill"]
        return bucket


class SlidingWindowRateLimiter(RateLimiter):
    """Sliding window rate limiter implementation."""
    
    def __init__(
        self,
        limit: int,
        window: float,  # seconds
        cache: Optional[Cache] = None,
        key_prefix: str = "rate_limit_sw",
        logger: Optional[Logger] = None,
    ):
        self.limit = limit
        self.window = window
        self.cache = cache
        self.key_prefix = key_prefix
        self.logger = logger
        
        # Local storage for when cache is not available
        self._local_windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str, weight: int = 1) -> bool:
        """Check if request is allowed."""
        try:
            return await self.consume(key, weight)
        except RateLimitExceeded:
            return False
    
    async def consume(self, key: str, weight: int = 1) -> bool:
        """Record request and check if allowed."""
        now = time.time()
        
        if self.cache:
            return await self._consume_with_cache(key, weight, now)
        else:
            return await self._consume_local(key, weight, now)
    
    async def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        
        if self.cache:
            timestamps = await self._get_timestamps_from_cache(key)
            timestamps = self._clean_timestamps(timestamps, now)
            current_usage = sum(1 for ts in timestamps if ts > now - self.window)
        else:
            async with self._lock:
                timestamps = self._local_windows.get(key, deque())
                self._clean_window(timestamps, now)
                current_usage = len(timestamps)
        
        return max(0, self.limit - current_usage)
    
    async def reset(self, key: str) -> None:
        """Reset rate limit for key."""
        if self.cache:
            cache_key = f"{self.key_prefix}:{key}"
            await self.cache.delete(cache_key)
        else:
            async with self._lock:
                self._local_windows.pop(key, None)
        
        if self.logger:
            self.logger.info(f"Rate limit reset for key: {key}")
    
    def _clean_window(self, timestamps: Deque[float], now: float) -> None:
        """Remove timestamps outside the window."""
        cutoff = now - self.window
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
    
    def _clean_timestamps(self, timestamps: list[float], now: float) -> list[float]:
        """Filter timestamps within window."""
        cutoff = now - self.window
        return [ts for ts in timestamps if ts >= cutoff]
    
    async def _consume_local(self, key: str, weight: int, now: float) -> bool:
        """Consume using local storage."""
        async with self._lock:
            timestamps = self._local_windows[key]
            self._clean_window(timestamps, now)
            
            if len(timestamps) + weight > self.limit:
                # Calculate retry after
                if timestamps:
                    oldest = timestamps[0]
                    retry_after = oldest + self.window - now
                else:
                    retry_after = 0
                
                raise RateLimitExceeded(
                    key=key,
                    limit=self.limit,
                    window=self.window,
                    retry_after=max(0, retry_after),
                )
            
            # Add new timestamps
            for _ in range(weight):
                timestamps.append(now)
            
            return True
    
    async def _consume_with_cache(self, key: str, weight: int, now: float) -> bool:
        """Consume using cache for distributed rate limiting."""
        cache_key = f"{self.key_prefix}:{key}"
        
        # Get current timestamps
        timestamps = await self._get_timestamps_from_cache(key)
        timestamps = self._clean_timestamps(timestamps, now)
        
        if len(timestamps) + weight > self.limit:
            # Calculate retry after
            if timestamps:
                oldest = min(timestamps)
                retry_after = oldest + self.window - now
            else:
                retry_after = 0
            
            raise RateLimitExceeded(
                key=key,
                limit=self.limit,
                window=self.window,
                retry_after=max(0, retry_after),
            )
        
        # Add new timestamps
        for _ in range(weight):
            timestamps.append(now)
        
        # Save updated timestamps
        await self.cache.set(cache_key, timestamps, int(self.window) + 1)
        
        return True
    
    async def _get_timestamps_from_cache(self, key: str) -> list[float]:
        """Get timestamps from cache."""
        cache_key = f"{self.key_prefix}:{key}"
        timestamps = await self.cache.get(cache_key)
        return timestamps or []


class RateLimiterDecorator:
    """Decorator for applying rate limiting to functions."""
    
    def __init__(
        self,
        rate_limiter: RateLimiter,
        key_func: Optional[Callable[..., str]] = None,
        weight: int = 1,
        raise_on_limit: bool = True,
    ):
        self.rate_limiter = rate_limiter
        self.key_func = key_func
        self.weight = weight
        self.raise_on_limit = raise_on_limit
    
    def __call__(self, func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """Decorate async function with rate limiting."""
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine rate limit key
            if self.key_func:
                key = self.key_func(*args, **kwargs)
            else:
                # Default to function name
                key = f"{func.__module__}.{func.__name__}"
            
            # Check rate limit
            try:
                await self.rate_limiter.consume(key, self.weight)
            except RateLimitExceeded as e:
                if self.raise_on_limit:
                    raise
                # Return None or some default value
                return None
            
            # Execute function
            return await func(*args, **kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.rate_limiter = self.rate_limiter
        
        return wrapper
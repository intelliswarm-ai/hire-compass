"""
Circuit breaker implementation for fault tolerance.

Implements the circuit breaker pattern to prevent cascading failures
and provide graceful degradation when external services fail.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

from src.shared.protocols import CircuitBreaker, Logger
from src.shared.types import Result, Success, Failure


T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerError(Exception):
    """Exception raised when circuit is open."""
    
    message: str
    circuit_name: str
    state: CircuitState
    last_failure_time: Optional[datetime] = None
    failure_count: int = 0


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: List[Tuple[datetime, CircuitState, CircuitState]] = field(
        default_factory=list
    )
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    success_threshold: int = 2  # successes needed in half-open to close
    timeout: Optional[float] = None  # call timeout
    error_types: Tuple[Type[Exception], ...] = (Exception,)
    exclude_types: Tuple[Type[Exception], ...] = ()
    
    def should_count_error(self, error: Exception) -> bool:
        """Check if error should count as failure."""
        # Check excluded types first
        if isinstance(error, self.exclude_types):
            return False
        
        # Check included types
        return isinstance(error, self.error_types)


class CircuitBreakerImpl(CircuitBreaker):
    """Circuit breaker implementation."""
    
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        timeout: Optional[float] = None,
        error_types: Optional[Tuple[Type[Exception], ...]] = None,
        exclude_types: Optional[Tuple[Type[Exception], ...]] = None,
        logger: Optional[Logger] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        self.name = name
        self.logger = logger
        self.on_state_change = on_state_change
        
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            timeout=timeout,
            error_types=error_types or (Exception,),
            exclude_types=exclude_types or (),
        )
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._half_open_start: Optional[datetime] = None
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self._state == CircuitState.OPEN
    
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self._state == CircuitState.CLOSED
    
    def is_half_open(self) -> bool:
        """Check if circuit is half-open."""
        return self._state == CircuitState.HALF_OPEN
    
    def get_failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count
    
    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self._stats
    
    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any
    ) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            await self._check_state_transition()
            
            # Check if circuit allows the call
            if self._state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                raise CircuitBreakerError(
                    message=f"Circuit breaker '{self.name}' is OPEN",
                    circuit_name=self.name,
                    state=self._state,
                    last_failure_time=self._last_failure_time,
                    failure_count=self._failure_count,
                )
        
        # Execute the function
        self._stats.total_calls += 1
        start_time = time.time()
        
        try:
            # Apply timeout if configured
            if self.config.timeout:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = await func(*args, **kwargs)
            
            # Record success
            await self._on_success()
            
            if self.logger:
                duration = time.time() - start_time
                self.logger.debug(
                    f"Circuit breaker '{self.name}' call succeeded",
                    duration_ms=duration * 1000,
                    state=self._state.value,
                )
            
            return result
            
        except Exception as error:
            # Record failure
            await self._on_failure(error)
            
            if self.logger:
                duration = time.time() - start_time
                self.logger.warning(
                    f"Circuit breaker '{self.name}' call failed",
                    duration_ms=duration * 1000,
                    state=self._state.value,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            
            raise
    
    async def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            # Check if recovery timeout has passed
            time_since_failure = (
                datetime.now() - self._last_failure_time
            ).total_seconds()
            
            if time_since_failure >= self.config.recovery_timeout:
                await self._transition_to_half_open()
    
    async def _on_success(self) -> None:
        """Handle successful call."""
        self._stats.successful_calls += 1
        
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
            
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
                self._stats.consecutive_failures = 0
    
    async def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        # Check if error should be counted
        if not self.config.should_count_error(error):
            return
        
        self._stats.failed_calls += 1
        self._last_failure_time = datetime.now()
        self._stats.last_failure_time = self._last_failure_time
        
        async with self._lock:
            self._failure_count += 1
            self._stats.consecutive_failures += 1
            
            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    await self._transition_to_open()
            
            elif self._state == CircuitState.HALF_OPEN:
                # Single failure in half-open goes back to open
                await self._transition_to_open()
    
    async def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        old_state = self._state
        self._state = CircuitState.OPEN
        self._record_transition(old_state, self._state)
        
        if self.logger:
            self.logger.warning(
                f"Circuit breaker '{self.name}' opened",
                failure_count=self._failure_count,
                threshold=self.config.failure_threshold,
            )
        
        if self.on_state_change:
            self.on_state_change(old_state, self._state)
    
    async def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._stats.consecutive_failures = 0
        self._record_transition(old_state, self._state)
        
        if self.logger:
            self.logger.info(
                f"Circuit breaker '{self.name}' closed",
                previous_state=old_state.value,
            )
        
        if self.on_state_change:
            self.on_state_change(old_state, self._state)
    
    async def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self._half_open_start = datetime.now()
        self._record_transition(old_state, self._state)
        
        if self.logger:
            self.logger.info(
                f"Circuit breaker '{self.name}' half-opened for testing",
                recovery_timeout=self.config.recovery_timeout,
            )
        
        if self.on_state_change:
            self.on_state_change(old_state, self._state)
    
    def _record_transition(self, from_state: CircuitState, to_state: CircuitState) -> None:
        """Record state transition."""
        self._stats.state_transitions.append(
            (datetime.now(), from_state, to_state)
        )
        
        # Keep only last 100 transitions
        if len(self._stats.state_transitions) > 100:
            self._stats.state_transitions = self._stats.state_transitions[-100:]
    
    async def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        async with self._lock:
            await self._transition_to_closed()
            self._last_failure_time = None
            self._half_open_start = None
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed state information."""
        info = {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "rejected_calls": self._stats.rejected_calls,
                "success_rate": self._stats.success_rate,
                "failure_rate": self._stats.failure_rate,
                "consecutive_failures": self._stats.consecutive_failures,
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            },
        }
        
        if self._last_failure_time:
            info["last_failure_time"] = self._last_failure_time.isoformat()
            info["time_since_failure"] = (
                datetime.now() - self._last_failure_time
            ).total_seconds()
        
        if self._half_open_start:
            info["half_open_duration"] = (
                datetime.now() - self._half_open_start
            ).total_seconds()
        
        return info


class CircuitBreakerDecorator:
    """Decorator for applying circuit breaker to functions."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        **kwargs: Any
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.kwargs = kwargs
        self._circuit_breakers: Dict[str, CircuitBreakerImpl] = {}
    
    def __call__(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """Decorate async function with circuit breaker."""
        circuit_name = self.name or f"{func.__module__}.{func.__name__}"
        
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get or create circuit breaker for this function
            if circuit_name not in self._circuit_breakers:
                self._circuit_breakers[circuit_name] = CircuitBreakerImpl(
                    name=circuit_name,
                    failure_threshold=self.failure_threshold,
                    recovery_timeout=self.recovery_timeout,
                    **self.kwargs
                )
            
            circuit_breaker = self._circuit_breakers[circuit_name]
            return await circuit_breaker.call(func, *args, **kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.circuit_breaker = lambda: self._circuit_breakers.get(circuit_name)
        
        return wrapper


# Convenience decorator
circuit_breaker = CircuitBreakerDecorator
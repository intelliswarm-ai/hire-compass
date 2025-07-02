"""
Event bus implementation for domain event publishing.

Provides a centralized event distribution mechanism with support for
async handlers, error handling, and event replay.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)
from uuid import UUID

from src.shared.protocols import (
    DomainEvent,
    EventHandler,
    EventPublisher,
    Logger,
    MessagePublisher,
)
from src.shared.types import Result, Success, Failure


TEvent = TypeVar("TEvent", bound=DomainEvent)
EventHandlerFunc = Callable[[DomainEvent], Awaitable[None]]


@dataclass
class HandlerRegistration:
    """Registration info for an event handler."""
    
    handler: Union[EventHandler, EventHandlerFunc]
    event_types: Set[Type[DomainEvent]]
    priority: int = 0
    retry_count: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    
    @property
    def handler_id(self) -> str:
        """Get unique handler identifier."""
        if hasattr(self.handler, "__name__"):
            return self.handler.__name__
        return self.handler.__class__.__name__


@dataclass
class EventContext:
    """Context for event processing."""
    
    event: DomainEvent
    handler_id: str
    attempt: int = 1
    start_time: float = field(default_factory=time.time)
    errors: List[Exception] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Get processing duration in seconds."""
        return time.time() - self.start_time


@dataclass
class EventBusMetrics:
    """Metrics for event bus operations."""
    
    events_published: int = 0
    events_handled: int = 0
    events_failed: int = 0
    handler_errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    processing_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def record_event_published(self) -> None:
        """Record event publication."""
        self.events_published += 1
    
    def record_event_handled(self, handler_id: str, duration: float) -> None:
        """Record successful event handling."""
        self.events_handled += 1
        self.processing_times[handler_id].append(duration)
    
    def record_event_failed(self, handler_id: str) -> None:
        """Record failed event handling."""
        self.events_failed += 1
        self.handler_errors[handler_id] += 1
    
    def get_average_processing_time(self, handler_id: str) -> Optional[float]:
        """Get average processing time for handler."""
        times = self.processing_times.get(handler_id, [])
        return sum(times) / len(times) if times else None


class EventBus(EventPublisher):
    """Event bus for domain event distribution."""
    
    def __init__(
        self,
        publisher: Optional[MessagePublisher] = None,
        logger: Optional[Logger] = None,
        enable_metrics: bool = True,
        enable_dead_letter: bool = True,
        max_dead_letter_size: int = 1000,
    ):
        self.publisher = publisher
        self.logger = logger
        self.enable_metrics = enable_metrics
        self.enable_dead_letter = enable_dead_letter
        self.max_dead_letter_size = max_dead_letter_size
        
        # Handler registry: event_type -> list of registrations
        self._handlers: Dict[Type[DomainEvent], List[HandlerRegistration]] = defaultdict(list)
        
        # Weak references to prevent memory leaks
        self._handler_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        
        # Metrics
        self.metrics = EventBusMetrics() if enable_metrics else None
        
        # Dead letter queue for failed events
        self._dead_letter_queue: List[Tuple[DomainEvent, EventContext]] = []
        
        # Event interceptors for cross-cutting concerns
        self._before_publish: List[Callable[[DomainEvent], Awaitable[None]]] = []
        self._after_publish: List[Callable[[DomainEvent], Awaitable[None]]] = []
    
    def register_handler(
        self,
        handler: Union[EventHandler, EventHandlerFunc],
        event_types: Optional[List[Type[DomainEvent]]] = None,
        priority: int = 0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
    ) -> None:
        """Register event handler."""
        # Determine event types
        if event_types is None:
            if hasattr(handler, "handles"):
                event_types = handler.handles()
            else:
                # Try to infer from type hints
                sig = inspect.signature(handler)
                params = list(sig.parameters.values())
                if params and hasattr(params[0].annotation, "__origin__"):
                    event_types = [params[0].annotation]
                else:
                    raise ValueError("Cannot determine event types for handler")
        
        # Create registration
        registration = HandlerRegistration(
            handler=handler,
            event_types=set(event_types),
            priority=priority,
            retry_count=retry_count,
            retry_delay=retry_delay,
            timeout=timeout,
        )
        
        # Register for each event type
        for event_type in event_types:
            self._handlers[event_type].append(registration)
            # Sort by priority (higher priority first)
            self._handlers[event_type].sort(key=lambda r: -r.priority)
        
        if self.logger:
            self.logger.info(
                f"Registered handler {registration.handler_id} "
                f"for {len(event_types)} event types"
            )
    
    def unregister_handler(
        self,
        handler: Union[EventHandler, EventHandlerFunc],
        event_types: Optional[List[Type[DomainEvent]]] = None,
    ) -> None:
        """Unregister event handler."""
        if event_types is None:
            # Remove from all event types
            for registrations in self._handlers.values():
                registrations[:] = [
                    r for r in registrations
                    if r.handler != handler
                ]
        else:
            # Remove from specific event types
            for event_type in event_types:
                self._handlers[event_type] = [
                    r for r in self._handlers[event_type]
                    if r.handler != handler
                ]
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish single event."""
        await self.publish_batch([event])
    
    async def publish_batch(self, events: List[DomainEvent]) -> None:
        """Publish multiple events."""
        for event in events:
            await self._publish_event(event)
    
    async def _publish_event(self, event: DomainEvent) -> None:
        """Internal event publication."""
        if self.logger:
            self.logger.debug(
                f"Publishing event: {event.event_type}",
                event_id=str(event.event_id),
                aggregate_id=str(event.aggregate_id),
            )
        
        # Run before-publish interceptors
        for interceptor in self._before_publish:
            try:
                await interceptor(event)
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Before-publish interceptor failed: {e}",
                        error=e
                    )
        
        # Record metric
        if self.metrics:
            self.metrics.record_event_published()
        
        # Get handlers for event type and its parent types
        handlers = self._get_handlers_for_event(event)
        
        # Handle event with each handler
        tasks = []
        for registration in handlers:
            task = asyncio.create_task(
                self._handle_event_safely(event, registration)
            )
            tasks.append(task)
        
        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Publish to external message queue if configured
        if self.publisher:
            try:
                await self.publisher.publish(
                    topic=f"events.{event.event_type.lower()}",
                    message={
                        "metadata": event.to_metadata(),
                        "payload": event.payload,
                    },
                    key=str(event.aggregate_id),
                )
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Failed to publish event to message queue: {e}",
                        error=e,
                        event_type=event.event_type,
                    )
        
        # Run after-publish interceptors
        for interceptor in self._after_publish:
            try:
                await interceptor(event)
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"After-publish interceptor failed: {e}",
                        error=e
                    )
    
    def _get_handlers_for_event(self, event: DomainEvent) -> List[HandlerRegistration]:
        """Get all handlers that should process this event."""
        handlers = []
        
        # Check exact type match
        event_type = type(event)
        if event_type in self._handlers:
            handlers.extend(self._handlers[event_type])
        
        # Check parent types (for inheritance)
        for base_type in event_type.__mro__[1:]:
            if base_type in self._handlers:
                handlers.extend(self._handlers[base_type])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_handlers = []
        for handler in handlers:
            if id(handler) not in seen:
                seen.add(id(handler))
                unique_handlers.append(handler)
        
        return unique_handlers
    
    async def _handle_event_safely(
        self,
        event: DomainEvent,
        registration: HandlerRegistration,
    ) -> None:
        """Handle event with error handling and retry logic."""
        context = EventContext(
            event=event,
            handler_id=registration.handler_id,
        )
        
        for attempt in range(1, registration.retry_count + 1):
            context.attempt = attempt
            
            try:
                # Apply timeout if configured
                if registration.timeout:
                    await asyncio.wait_for(
                        self._invoke_handler(registration.handler, event),
                        timeout=registration.timeout,
                    )
                else:
                    await self._invoke_handler(registration.handler, event)
                
                # Success - record metrics and return
                if self.metrics:
                    self.metrics.record_event_handled(
                        registration.handler_id,
                        context.duration,
                    )
                
                if self.logger:
                    self.logger.debug(
                        f"Event handled successfully",
                        handler=registration.handler_id,
                        event_type=event.event_type,
                        attempt=attempt,
                        duration=context.duration,
                    )
                
                return
                
            except asyncio.TimeoutError:
                error = TimeoutError(
                    f"Handler {registration.handler_id} timed out after "
                    f"{registration.timeout} seconds"
                )
                context.errors.append(error)
                
                if self.logger:
                    self.logger.warning(
                        f"Handler timeout",
                        handler=registration.handler_id,
                        event_type=event.event_type,
                        attempt=attempt,
                        timeout=registration.timeout,
                    )
                
            except Exception as e:
                context.errors.append(e)
                
                if self.logger:
                    self.logger.error(
                        f"Handler error",
                        handler=registration.handler_id,
                        event_type=event.event_type,
                        attempt=attempt,
                        error=e,
                    )
            
            # Retry delay (except for last attempt)
            if attempt < registration.retry_count:
                await asyncio.sleep(registration.retry_delay * attempt)
        
        # All attempts failed - record failure
        if self.metrics:
            self.metrics.record_event_failed(registration.handler_id)
        
        # Add to dead letter queue if enabled
        if self.enable_dead_letter:
            self._add_to_dead_letter(event, context)
        
        if self.logger:
            self.logger.error(
                f"Handler failed after {registration.retry_count} attempts",
                handler=registration.handler_id,
                event_type=event.event_type,
                errors=[str(e) for e in context.errors],
            )
    
    async def _invoke_handler(
        self,
        handler: Union[EventHandler, EventHandlerFunc],
        event: DomainEvent,
    ) -> None:
        """Invoke event handler."""
        if hasattr(handler, "handle"):
            # EventHandler protocol
            await handler.handle(event)
        else:
            # Function handler
            await handler(event)
    
    def _add_to_dead_letter(self, event: DomainEvent, context: EventContext) -> None:
        """Add failed event to dead letter queue."""
        self._dead_letter_queue.append((event, context))
        
        # Trim queue if it exceeds max size (FIFO)
        if len(self._dead_letter_queue) > self.max_dead_letter_size:
            self._dead_letter_queue.pop(0)
    
    def get_dead_letter_events(self) -> List[Tuple[DomainEvent, EventContext]]:
        """Get events from dead letter queue."""
        return self._dead_letter_queue.copy()
    
    def clear_dead_letter_queue(self) -> None:
        """Clear dead letter queue."""
        self._dead_letter_queue.clear()
    
    async def replay_dead_letter_event(self, index: int) -> bool:
        """Replay a specific event from dead letter queue."""
        if 0 <= index < len(self._dead_letter_queue):
            event, _ = self._dead_letter_queue[index]
            try:
                await self.publish(event)
                # Remove from dead letter on success
                self._dead_letter_queue.pop(index)
                return True
            except Exception:
                return False
        return False
    
    def add_before_publish_interceptor(
        self,
        interceptor: Callable[[DomainEvent], Awaitable[None]],
    ) -> None:
        """Add interceptor to run before event publication."""
        self._before_publish.append(interceptor)
    
    def add_after_publish_interceptor(
        self,
        interceptor: Callable[[DomainEvent], Awaitable[None]],
    ) -> None:
        """Add interceptor to run after event publication."""
        self._after_publish.append(interceptor)
    
    def get_metrics(self) -> Optional[EventBusMetrics]:
        """Get event bus metrics."""
        return self.metrics
    
    def get_registered_handlers(self) -> Dict[str, List[str]]:
        """Get summary of registered handlers."""
        summary = {}
        for event_type, registrations in self._handlers.items():
            event_name = event_type.__name__
            handler_names = [r.handler_id for r in registrations]
            summary[event_name] = handler_names
        return summary
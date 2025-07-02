"""
Protocol definitions for dependency inversion.

This module defines the contracts (interfaces) that components must implement,
following the Dependency Inversion Principle from SOLID.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)
from datetime import datetime
from uuid import UUID

from .types import (
    T,
    TEntity,
    TAggregate,
    AuditInfo,
    CommandMetadata,
    EventMetadata,
    EventPayload,
    FilterExpression,
    PaginatedResult,
    PaginationParams,
    QueryMetadata,
    Result,
    TimeRange,
)


# Repository protocols
@runtime_checkable
class Repository(Protocol[TEntity]):
    """Base repository protocol for data access."""
    
    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> Optional[TEntity]:
        """Retrieve entity by ID."""
        ...
    
    @abstractmethod
    async def exists(self, entity_id: UUID) -> bool:
        """Check if entity exists."""
        ...
    
    @abstractmethod
    async def save(self, entity: TEntity) -> None:
        """Save entity to storage."""
        ...
    
    @abstractmethod
    async def delete(self, entity_id: UUID) -> None:
        """Delete entity from storage."""
        ...


@runtime_checkable
class ReadRepository(Protocol[TEntity]):
    """Read-only repository protocol."""
    
    @abstractmethod
    async def get_by_id(self, entity_id: UUID) -> Optional[TEntity]:
        """Retrieve entity by ID."""
        ...
    
    @abstractmethod
    async def get_many(self, entity_ids: List[UUID]) -> List[TEntity]:
        """Retrieve multiple entities by IDs."""
        ...
    
    @abstractmethod
    async def find(
        self,
        filter_expression: Optional[FilterExpression] = None,
        pagination: Optional[PaginationParams] = None,
    ) -> PaginatedResult[TEntity]:
        """Find entities with filtering and pagination."""
        ...
    
    @abstractmethod
    async def count(self, filter_expression: Optional[FilterExpression] = None) -> int:
        """Count entities matching filter."""
        ...


@runtime_checkable
class WriteRepository(Protocol[TEntity]):
    """Write-only repository protocol."""
    
    @abstractmethod
    async def save(self, entity: TEntity) -> None:
        """Save entity to storage."""
        ...
    
    @abstractmethod
    async def save_many(self, entities: List[TEntity]) -> None:
        """Save multiple entities."""
        ...
    
    @abstractmethod
    async def delete(self, entity_id: UUID) -> None:
        """Delete entity from storage."""
        ...
    
    @abstractmethod
    async def delete_many(self, entity_ids: List[UUID]) -> None:
        """Delete multiple entities."""
        ...


# Unit of Work protocol
@runtime_checkable
class UnitOfWork(Protocol):
    """Unit of Work pattern for transactional consistency."""
    
    @abstractmethod
    async def __aenter__(self) -> UnitOfWork:
        """Begin unit of work."""
        ...
    
    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End unit of work."""
        ...
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit all changes."""
        ...
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback all changes."""
        ...


# Event protocols
@runtime_checkable
class DomainEvent(Protocol):
    """Domain event protocol."""
    
    @property
    @abstractmethod
    def event_type(self) -> str:
        """Event type identifier."""
        ...
    
    @property
    @abstractmethod
    def aggregate_id(self) -> UUID:
        """ID of the aggregate that raised the event."""
        ...
    
    @property
    @abstractmethod
    def occurred_at(self) -> datetime:
        """When the event occurred."""
        ...
    
    @property
    @abstractmethod
    def payload(self) -> EventPayload:
        """Event data payload."""
        ...
    
    @abstractmethod
    def to_metadata(self) -> EventMetadata:
        """Convert to event metadata."""
        ...


@runtime_checkable
class EventStore(Protocol):
    """Event store for event sourcing."""
    
    @abstractmethod
    async def append(self, stream_id: str, events: List[DomainEvent]) -> None:
        """Append events to stream."""
        ...
    
    @abstractmethod
    async def get_events(
        self,
        stream_id: str,
        from_version: Optional[int] = None,
        to_version: Optional[int] = None,
    ) -> List[DomainEvent]:
        """Get events from stream."""
        ...
    
    @abstractmethod
    async def get_events_by_type(
        self,
        event_type: str,
        time_range: Optional[TimeRange] = None,
    ) -> List[DomainEvent]:
        """Get events by type."""
        ...


@runtime_checkable
class EventPublisher(Protocol):
    """Event publisher for domain events."""
    
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish single event."""
        ...
    
    @abstractmethod
    async def publish_batch(self, events: List[DomainEvent]) -> None:
        """Publish multiple events."""
        ...


@runtime_checkable
class EventHandler(Protocol):
    """Handler for domain events."""
    
    @abstractmethod
    def handles(self) -> List[Type[DomainEvent]]:
        """Event types this handler processes."""
        ...
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle the event."""
        ...


# Command and Query protocols
@runtime_checkable
class Command(Protocol):
    """Command protocol for CQRS."""
    
    @property
    @abstractmethod
    def command_type(self) -> str:
        """Command type identifier."""
        ...
    
    @abstractmethod
    def to_metadata(self) -> CommandMetadata:
        """Convert to command metadata."""
        ...


@runtime_checkable
class CommandHandler(Protocol[T]):
    """Handler for commands."""
    
    @abstractmethod
    def handles(self) -> Type[Command]:
        """Command type this handler processes."""
        ...
    
    @abstractmethod
    async def handle(self, command: Command) -> Result[T]:
        """Handle the command."""
        ...


@runtime_checkable
class Query(Protocol):
    """Query protocol for CQRS."""
    
    @property
    @abstractmethod
    def query_type(self) -> str:
        """Query type identifier."""
        ...
    
    @abstractmethod
    def to_metadata(self) -> QueryMetadata:
        """Convert to query metadata."""
        ...


@runtime_checkable
class QueryHandler(Protocol[T]):
    """Handler for queries."""
    
    @abstractmethod
    def handles(self) -> Type[Query]:
        """Query type this handler processes."""
        ...
    
    @abstractmethod
    async def handle(self, query: Query) -> Result[T]:
        """Handle the query."""
        ...


# Mediator protocol
@runtime_checkable
class Mediator(Protocol):
    """Mediator for command and query handling."""
    
    @abstractmethod
    async def send_command(self, command: Command) -> Result[Any]:
        """Send command to handler."""
        ...
    
    @abstractmethod
    async def send_query(self, query: Query) -> Result[Any]:
        """Send query to handler."""
        ...
    
    @abstractmethod
    async def publish_event(self, event: DomainEvent) -> None:
        """Publish domain event."""
        ...


# Caching protocols
@runtime_checkable
class Cache(Protocol):
    """Cache abstraction."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL in seconds."""
        ...
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache."""
        ...
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        ...


# Message queue protocols
@runtime_checkable
class MessagePublisher(Protocol):
    """Message publisher for async messaging."""
    
    @abstractmethod
    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
    ) -> None:
        """Publish message to topic."""
        ...


@runtime_checkable
class MessageConsumer(Protocol):
    """Message consumer for async messaging."""
    
    @abstractmethod
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
        group_id: Optional[str] = None,
    ) -> None:
        """Subscribe to topic with handler."""
        ...
    
    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from topic."""
        ...


# Transaction protocols
@runtime_checkable
class TransactionManager(Protocol):
    """Transaction management."""
    
    @abstractmethod
    async def begin(self) -> None:
        """Begin transaction."""
        ...
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit transaction."""
        ...
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback transaction."""
        ...


# Health check protocol
@runtime_checkable
class HealthCheck(Protocol):
    """Health check for services."""
    
    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if service is healthy."""
        ...
    
    @abstractmethod
    async def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        ...


# Circuit breaker protocol
@runtime_checkable
class CircuitBreaker(Protocol):
    """Circuit breaker for fault tolerance."""
    
    @abstractmethod
    async def call(self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection."""
        ...
    
    @abstractmethod
    def is_open(self) -> bool:
        """Check if circuit is open."""
        ...
    
    @abstractmethod
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        ...
    
    @abstractmethod
    def get_failure_count(self) -> int:
        """Get current failure count."""
        ...


# Rate limiter protocol
@runtime_checkable
class RateLimiter(Protocol):
    """Rate limiter for API protection."""
    
    @abstractmethod
    async def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        ...
    
    @abstractmethod
    async def consume(self, key: str, tokens: int = 1) -> bool:
        """Consume tokens from bucket."""
        ...
    
    @abstractmethod
    async def get_remaining(self, key: str) -> int:
        """Get remaining tokens."""
        ...


# Logger protocol
@runtime_checkable
class Logger(Protocol):
    """Structured logger abstraction."""
    
    @abstractmethod
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        ...
    
    @abstractmethod
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        ...
    
    @abstractmethod
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        ...
    
    @abstractmethod
    def error(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log error message."""
        ...
    
    @abstractmethod
    def with_context(self, **context: Any) -> Logger:
        """Create logger with additional context."""
        ...


# Metrics collector protocol
@runtime_checkable
class MetricsCollector(Protocol):
    """Metrics collection abstraction."""
    
    @abstractmethod
    def increment(self, metric: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment counter metric."""
        ...
    
    @abstractmethod
    def gauge(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Set gauge metric."""
        ...
    
    @abstractmethod
    def histogram(self, metric: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record histogram metric."""
        ...
    
    @abstractmethod
    def timing(self, metric: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record timing metric."""
        ...
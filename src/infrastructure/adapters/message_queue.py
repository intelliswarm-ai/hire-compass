"""
Message queue adapter implementations.

Provides Kafka and in-memory message queue implementations for async messaging.
Supports both publish/subscribe and point-to-point messaging patterns.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
from kafka.errors import KafkaConnectionError

from src.shared.protocols import Logger, MessageConsumer, MessagePublisher


@dataclass
class Message:
    """Message container."""
    
    topic: str
    key: Optional[str]
    value: Dict[str, Any]
    headers: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.now)
    partition: Optional[int] = None
    offset: Optional[int] = None
    
    def to_bytes(self) -> bytes:
        """Serialize message value to bytes."""
        return json.dumps(self.value).encode("utf-8")
    
    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> Message:
        """Deserialize message from bytes."""
        value = json.loads(data.decode("utf-8"))
        return cls(value=value, **kwargs)


@dataclass
class ConsumerGroup:
    """Consumer group for message distribution."""
    
    group_id: str
    consumers: Set[str] = field(default_factory=set)
    current_consumer_index: int = 0
    
    def add_consumer(self, consumer_id: str) -> None:
        """Add consumer to group."""
        self.consumers.add(consumer_id)
    
    def remove_consumer(self, consumer_id: str) -> None:
        """Remove consumer from group."""
        self.consumers.discard(consumer_id)
    
    def get_next_consumer(self) -> Optional[str]:
        """Get next consumer in round-robin fashion."""
        if not self.consumers:
            return None
        
        consumer_list = sorted(self.consumers)  # For consistency
        consumer = consumer_list[self.current_consumer_index % len(consumer_list)]
        self.current_consumer_index += 1
        return consumer


class KafkaMessageQueue(MessagePublisher, MessageConsumer):
    """Kafka-based message queue implementation."""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        client_id: Optional[str] = None,
        producer_config: Optional[Dict[str, Any]] = None,
        consumer_config: Optional[Dict[str, Any]] = None,
        logger: Optional[Logger] = None,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id or f"hr-matcher-{uuid.uuid4().hex[:8]}"
        self.producer_config = producer_config or {}
        self.consumer_config = consumer_config or {}
        self.logger = logger
        
        self._producer: Optional[AIOKafkaProducer] = None
        self._consumers: Dict[str, AIOKafkaConsumer] = {}
        self._consumer_tasks: Dict[str, asyncio.Task] = {}
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = {}
        self._running = False
    
    async def start(self) -> None:
        """Start message queue connections."""
        self._running = True
        await self._ensure_producer()
        
        if self.logger:
            self.logger.info(
                f"Kafka message queue started",
                client_id=self.client_id,
                bootstrap_servers=self.bootstrap_servers,
            )
    
    async def stop(self) -> None:
        """Stop message queue connections."""
        self._running = False
        
        # Cancel consumer tasks
        for task in self._consumer_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._consumer_tasks:
            await asyncio.gather(
                *self._consumer_tasks.values(),
                return_exceptions=True
            )
        
        # Stop consumers
        for consumer in self._consumers.values():
            await consumer.stop()
        
        # Stop producer
        if self._producer:
            await self._producer.stop()
        
        self._consumers.clear()
        self._consumer_tasks.clear()
        self._handlers.clear()
        self._producer = None
        
        if self.logger:
            self.logger.info("Kafka message queue stopped")
    
    async def _ensure_producer(self) -> AIOKafkaProducer:
        """Ensure producer is initialized."""
        if self._producer is None:
            config = {
                "bootstrap_servers": self.bootstrap_servers,
                "client_id": f"{self.client_id}-producer",
                "enable_idempotence": True,
                "acks": "all",
                "retries": 3,
                "max_in_flight_requests_per_connection": 5,
                "compression_type": "gzip",
                **self.producer_config,
            }
            
            self._producer = AIOKafkaProducer(**config)
            await self._producer.start()
        
        return self._producer
    
    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish message to topic."""
        if not self._running:
            raise RuntimeError("Message queue not started")
        
        producer = await self._ensure_producer()
        
        try:
            # Prepare message
            value = json.dumps(message).encode("utf-8")
            key_bytes = key.encode("utf-8") if key else None
            
            # Convert headers
            kafka_headers = [
                (k, v.encode("utf-8"))
                for k, v in (headers or {}).items()
            ]
            
            # Send message
            metadata = await producer.send_and_wait(
                topic=topic,
                value=value,
                key=key_bytes,
                headers=kafka_headers,
            )
            
            if self.logger:
                self.logger.debug(
                    f"Message published",
                    topic=topic,
                    partition=metadata.partition,
                    offset=metadata.offset,
                    key=key,
                )
            
        except KafkaError as e:
            if self.logger:
                self.logger.error(
                    f"Failed to publish message",
                    topic=topic,
                    error=e,
                )
            raise
    
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
        group_id: Optional[str] = None,
        from_beginning: bool = False,
    ) -> None:
        """Subscribe to topic with handler."""
        if not self._running:
            raise RuntimeError("Message queue not started")
        
        consumer_id = f"{topic}-{group_id or 'default'}"
        
        # Store handler
        self._handlers[consumer_id] = handler
        
        # Create consumer if not exists
        if consumer_id not in self._consumers:
            config = {
                "bootstrap_servers": self.bootstrap_servers,
                "client_id": f"{self.client_id}-consumer-{consumer_id}",
                "group_id": group_id,
                "enable_auto_commit": False,
                "auto_offset_reset": "earliest" if from_beginning else "latest",
                **self.consumer_config,
            }
            
            consumer = AIOKafkaConsumer(topic, **config)
            await consumer.start()
            self._consumers[consumer_id] = consumer
            
            # Start consumer task
            task = asyncio.create_task(
                self._consume_messages(consumer_id, consumer)
            )
            self._consumer_tasks[consumer_id] = task
        
        if self.logger:
            self.logger.info(
                f"Subscribed to topic",
                topic=topic,
                group_id=group_id,
            )
    
    async def unsubscribe(self, topic: str, group_id: Optional[str] = None) -> None:
        """Unsubscribe from topic."""
        consumer_id = f"{topic}-{group_id or 'default'}"
        
        # Cancel consumer task
        if consumer_id in self._consumer_tasks:
            self._consumer_tasks[consumer_id].cancel()
            try:
                await self._consumer_tasks[consumer_id]
            except asyncio.CancelledError:
                pass
            del self._consumer_tasks[consumer_id]
        
        # Stop consumer
        if consumer_id in self._consumers:
            await self._consumers[consumer_id].stop()
            del self._consumers[consumer_id]
        
        # Remove handler
        self._handlers.pop(consumer_id, None)
        
        if self.logger:
            self.logger.info(
                f"Unsubscribed from topic",
                topic=topic,
                group_id=group_id,
            )
    
    async def _consume_messages(
        self,
        consumer_id: str,
        consumer: AIOKafkaConsumer,
    ) -> None:
        """Consume messages from Kafka."""
        handler = self._handlers[consumer_id]
        
        while self._running:
            try:
                # Fetch messages
                messages = await consumer.getmany(
                    timeout_ms=1000,
                    max_records=100,
                )
                
                for topic_partition, records in messages.items():
                    for record in records:
                        try:
                            # Parse message
                            message_data = json.loads(
                                record.value.decode("utf-8")
                            )
                            
                            # Call handler
                            await handler(message_data)
                            
                            # Commit offset
                            await consumer.commit()
                            
                        except Exception as e:
                            if self.logger:
                                self.logger.error(
                                    f"Error handling message",
                                    consumer_id=consumer_id,
                                    topic=topic_partition.topic,
                                    partition=topic_partition.partition,
                                    offset=record.offset,
                                    error=e,
                                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Consumer error",
                        consumer_id=consumer_id,
                        error=e,
                    )
                # Brief pause before retry
                await asyncio.sleep(1)


class InMemoryMessageQueue(MessagePublisher, MessageConsumer):
    """In-memory message queue implementation for testing."""
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        
        # Topic -> Queue of messages
        self._queues: Dict[str, Deque[Message]] = defaultdict(
            lambda: deque(maxlen=max_queue_size)
        )
        
        # Topic -> Group ID -> Consumer Group
        self._consumer_groups: Dict[str, Dict[str, ConsumerGroup]] = defaultdict(dict)
        
        # Consumer ID -> (handler, task)
        self._consumers: Dict[str, Tuple[
            Callable[[Dict[str, Any]], Awaitable[None]],
            Optional[asyncio.Task]
        ]] = {}
        
        # Topic -> Set of broadcast handlers (no group)
        self._broadcast_handlers: Dict[str, Set[
            Callable[[Dict[str, Any]], Awaitable[None]]
        ]] = defaultdict(set)
        
        self._running = False
        self._message_counter = 0
    
    async def start(self) -> None:
        """Start message queue."""
        self._running = True
    
    async def stop(self) -> None:
        """Stop message queue."""
        self._running = False
        
        # Cancel all consumer tasks
        for consumer_id, (_, task) in self._consumers.items():
            if task:
                task.cancel()
        
        # Wait for tasks to complete
        tasks = [task for _, task in self._consumers.values() if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Clear state
        self._queues.clear()
        self._consumer_groups.clear()
        self._consumers.clear()
        self._broadcast_handlers.clear()
    
    async def publish(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish message to topic."""
        if not self._running:
            raise RuntimeError("Message queue not started")
        
        # Create message
        msg = Message(
            topic=topic,
            key=key,
            value=message,
            headers=headers or {},
            offset=self._message_counter,
        )
        self._message_counter += 1
        
        # Add to queue
        self._queues[topic].append(msg)
        
        # Notify broadcast handlers immediately
        for handler in self._broadcast_handlers.get(topic, set()):
            asyncio.create_task(self._handle_message(handler, msg))
        
        # Notify consumer groups
        for group_id, group in self._consumer_groups.get(topic, {}).items():
            consumer_id = group.get_next_consumer()
            if consumer_id and consumer_id in self._consumers:
                handler, _ = self._consumers[consumer_id]
                asyncio.create_task(self._handle_message(handler, msg))
    
    async def subscribe(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
        group_id: Optional[str] = None,
        from_beginning: bool = False,
    ) -> None:
        """Subscribe to topic with handler."""
        if not self._running:
            raise RuntimeError("Message queue not started")
        
        if group_id:
            # Consumer group subscription
            consumer_id = f"{topic}-{group_id}-{uuid.uuid4().hex[:8]}"
            
            # Create or get consumer group
            if group_id not in self._consumer_groups[topic]:
                self._consumer_groups[topic][group_id] = ConsumerGroup(group_id)
            
            group = self._consumer_groups[topic][group_id]
            group.add_consumer(consumer_id)
            
            # Store consumer
            self._consumers[consumer_id] = (handler, None)
            
            # Process existing messages if from_beginning
            if from_beginning:
                for msg in self._queues[topic]:
                    await self._handle_message(handler, msg)
        else:
            # Broadcast subscription (all messages)
            self._broadcast_handlers[topic].add(handler)
            
            # Process existing messages if from_beginning
            if from_beginning:
                for msg in self._queues[topic]:
                    await self._handle_message(handler, msg)
    
    async def unsubscribe(
        self,
        topic: str,
        handler: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        group_id: Optional[str] = None,
    ) -> None:
        """Unsubscribe from topic."""
        if group_id:
            # Remove from consumer group
            if topic in self._consumer_groups and group_id in self._consumer_groups[topic]:
                group = self._consumer_groups[topic][group_id]
                
                # Find and remove consumer
                for consumer_id in list(self._consumers.keys()):
                    if consumer_id.startswith(f"{topic}-{group_id}-"):
                        group.remove_consumer(consumer_id)
                        del self._consumers[consumer_id]
                
                # Remove empty group
                if not group.consumers:
                    del self._consumer_groups[topic][group_id]
        else:
            # Remove broadcast handler
            if handler and topic in self._broadcast_handlers:
                self._broadcast_handlers[topic].discard(handler)
    
    async def _handle_message(
        self,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
        message: Message,
    ) -> None:
        """Handle message with error handling."""
        try:
            await handler(message.value)
        except Exception:
            # Silently ignore errors in test implementation
            pass
    
    def get_queue_size(self, topic: str) -> int:
        """Get current queue size for topic."""
        return len(self._queues.get(topic, []))
    
    def clear_topic(self, topic: str) -> None:
        """Clear all messages from topic."""
        if topic in self._queues:
            self._queues[topic].clear()
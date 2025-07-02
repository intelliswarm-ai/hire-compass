"""
Dependency injection container using dependency-injector.

This module implements the Inversion of Control (IoC) container that manages
all dependencies in the application, ensuring proper separation of concerns
and making the system highly testable and maintainable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from dependency_injector import containers, providers
from dependency_injector.providers import Configuration, Factory, Singleton

from src.infrastructure.config.settings import Settings
from src.infrastructure.adapters.cache import RedisCache, InMemoryCache
from src.infrastructure.adapters.event_bus import EventBus
from src.infrastructure.adapters.message_queue import KafkaMessageQueue, InMemoryMessageQueue
from src.infrastructure.adapters.metrics import PrometheusMetrics, ConsoleMetrics
from src.infrastructure.adapters.logger import StructuredLogger
from src.infrastructure.adapters.vector_store import ChromaVectorStore
from src.infrastructure.adapters.llm import OllamaLLM
from src.infrastructure.adapters.circuit_breaker import CircuitBreakerImpl
from src.infrastructure.adapters.rate_limiter import TokenBucketRateLimiter

from src.infrastructure.repositories.resume_repository import (
    ResumeRepository,
    ResumeReadRepository,
    ResumeWriteRepository,
)
from src.infrastructure.repositories.position_repository import (
    PositionRepository,
    PositionReadRepository,
    PositionWriteRepository,
)
from src.infrastructure.repositories.match_repository import MatchRepository
from src.infrastructure.repositories.event_store import EventStoreImpl

from src.application.use_cases.parse_resume import ParseResumeUseCase
from src.application.use_cases.parse_position import ParsePositionUseCase
from src.application.use_cases.match_resume_to_position import MatchResumeToPositionUseCase
from src.application.use_cases.batch_match import BatchMatchUseCase
from src.application.use_cases.research_salary import ResearchSalaryUseCase
from src.application.use_cases.analyze_aspirations import AnalyzeAspirationsUseCase

from src.domain.services.matching_service import MatchingService
from src.domain.services.scoring_service import ScoringService
from src.domain.services.skill_matcher import SkillMatcher

from src.presentation.api.health_check import HealthCheckService


class Container(containers.DeclarativeContainer):
    """Main dependency injection container."""
    
    # Configuration
    config = Configuration()
    settings = providers.Singleton(Settings)
    
    # Infrastructure - Logging
    logger = providers.Singleton(
        StructuredLogger,
        name="hr_matcher",
        level=config.logging.level,
        json_format=config.logging.json_format,
    )
    
    # Infrastructure - Metrics
    metrics = providers.Selector(
        config.metrics.provider,
        prometheus=providers.Singleton(
            PrometheusMetrics,
            port=config.metrics.prometheus_port,
        ),
        console=providers.Singleton(ConsoleMetrics),
    )
    
    # Infrastructure - Cache
    cache = providers.Selector(
        config.cache.provider,
        redis=providers.Singleton(
            RedisCache,
            host=config.cache.redis_host,
            port=config.cache.redis_port,
            db=config.cache.redis_db,
            password=config.cache.redis_password,
            default_ttl=config.cache.default_ttl,
            logger=logger,
        ),
        memory=providers.Singleton(
            InMemoryCache,
            max_size=config.cache.memory_max_size,
            default_ttl=config.cache.default_ttl,
        ),
    )
    
    # Infrastructure - Message Queue
    message_queue = providers.Selector(
        config.messaging.provider,
        kafka=providers.Singleton(
            KafkaMessageQueue,
            bootstrap_servers=config.messaging.kafka_bootstrap_servers,
            logger=logger,
        ),
        memory=providers.Singleton(InMemoryMessageQueue),
    )
    
    # Infrastructure - Event Bus
    event_bus = providers.Singleton(
        EventBus,
        publisher=message_queue,
        logger=logger,
    )
    
    # Infrastructure - Event Store
    event_store = providers.Singleton(
        EventStoreImpl,
        connection_string=config.database.event_store_connection,
        logger=logger,
    )
    
    # Infrastructure - Vector Store
    vector_store = providers.Singleton(
        ChromaVectorStore,
        persist_directory=config.vector_store.persist_directory,
        collection_name=config.vector_store.collection_name,
        embedding_model=config.vector_store.embedding_model,
        logger=logger,
    )
    
    # Infrastructure - LLM
    llm = providers.Singleton(
        OllamaLLM,
        base_url=config.llm.ollama_base_url,
        model=config.llm.model,
        temperature=config.llm.temperature,
        timeout=config.llm.timeout,
        logger=logger,
    )
    
    # Infrastructure - Circuit Breaker
    circuit_breaker_factory = providers.Factory(
        CircuitBreakerImpl,
        failure_threshold=config.resilience.circuit_breaker_threshold,
        recovery_timeout=config.resilience.circuit_breaker_timeout,
        logger=logger,
    )
    
    # Infrastructure - Rate Limiter
    rate_limiter = providers.Singleton(
        TokenBucketRateLimiter,
        rate=config.resilience.rate_limit_requests,
        burst=config.resilience.rate_limit_burst,
        cache=cache,
        logger=logger,
    )
    
    # Repositories
    resume_repository = providers.Singleton(
        ResumeRepository,
        vector_store=vector_store,
        event_store=event_store,
        cache=cache,
        logger=logger,
    )
    
    resume_read_repository = providers.Singleton(
        ResumeReadRepository,
        vector_store=vector_store,
        cache=cache,
        logger=logger,
    )
    
    resume_write_repository = providers.Singleton(
        ResumeWriteRepository,
        vector_store=vector_store,
        event_store=event_store,
        event_bus=event_bus,
        logger=logger,
    )
    
    position_repository = providers.Singleton(
        PositionRepository,
        vector_store=vector_store,
        event_store=event_store,
        cache=cache,
        logger=logger,
    )
    
    position_read_repository = providers.Singleton(
        PositionReadRepository,
        vector_store=vector_store,
        cache=cache,
        logger=logger,
    )
    
    position_write_repository = providers.Singleton(
        PositionWriteRepository,
        vector_store=vector_store,
        event_store=event_store,
        event_bus=event_bus,
        logger=logger,
    )
    
    match_repository = providers.Singleton(
        MatchRepository,
        database=config.database.connection_string,
        cache=cache,
        logger=logger,
    )
    
    # Domain Services
    skill_matcher = providers.Singleton(
        SkillMatcher,
        embedding_model=config.vector_store.embedding_model,
        cache=cache,
        logger=logger,
    )
    
    scoring_service = providers.Singleton(
        ScoringService,
        skill_matcher=skill_matcher,
        weights=config.matching.score_weights,
        logger=logger,
    )
    
    matching_service = providers.Singleton(
        MatchingService,
        scoring_service=scoring_service,
        skill_matcher=skill_matcher,
        logger=logger,
    )
    
    # Application Use Cases
    parse_resume_use_case = providers.Factory(
        ParseResumeUseCase,
        resume_repository=resume_write_repository,
        llm=llm,
        event_bus=event_bus,
        circuit_breaker=circuit_breaker_factory,
        logger=logger,
    )
    
    parse_position_use_case = providers.Factory(
        ParsePositionUseCase,
        position_repository=position_write_repository,
        llm=llm,
        event_bus=event_bus,
        circuit_breaker=circuit_breaker_factory,
        logger=logger,
    )
    
    match_resume_to_position_use_case = providers.Factory(
        MatchResumeToPositionUseCase,
        resume_repository=resume_read_repository,
        position_repository=position_read_repository,
        match_repository=match_repository,
        matching_service=matching_service,
        event_bus=event_bus,
        cache=cache,
        logger=logger,
    )
    
    batch_match_use_case = providers.Factory(
        BatchMatchUseCase,
        resume_repository=resume_read_repository,
        position_repository=position_read_repository,
        match_repository=match_repository,
        matching_service=matching_service,
        event_bus=event_bus,
        cache=cache,
        logger=logger,
        max_workers=config.performance.max_workers,
    )
    
    research_salary_use_case = providers.Factory(
        ResearchSalaryUseCase,
        cache=cache,
        circuit_breaker=circuit_breaker_factory,
        logger=logger,
        timeout=config.external_apis.salary_research_timeout,
    )
    
    analyze_aspirations_use_case = providers.Factory(
        AnalyzeAspirationsUseCase,
        resume_repository=resume_read_repository,
        llm=llm,
        event_bus=event_bus,
        cache=cache,
        logger=logger,
    )
    
    # Presentation Services
    health_check_service = providers.Singleton(
        HealthCheckService,
        vector_store=vector_store,
        cache=cache,
        message_queue=message_queue,
        llm=llm,
        logger=logger,
    )


class TestContainer(Container):
    """Test container with mock implementations."""
    
    # Override with test doubles
    cache = providers.Singleton(
        InMemoryCache,
        max_size=1000,
        default_ttl=60,
    )
    
    message_queue = providers.Singleton(InMemoryMessageQueue)
    
    # Use in-memory implementations for tests
    vector_store = providers.Singleton(
        ChromaVectorStore,
        persist_directory=":memory:",
        collection_name="test_collection",
        embedding_model="test",
    )


def create_container(
    config_path: Optional[str] = None,
    environment: Optional[str] = None,
) -> Container:
    """
    Create and configure the dependency injection container.
    
    Args:
        config_path: Path to configuration file
        environment: Environment name (development, staging, production, test)
    
    Returns:
        Configured container instance
    """
    # Determine environment
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    # Create appropriate container
    if env == "test":
        container = TestContainer()
    else:
        container = Container()
    
    # Load configuration
    if config_path:
        container.config.from_yaml(config_path)
    else:
        # Load from environment-specific file
        config_file = f"config.{env}.yaml"
        if os.path.exists(config_file):
            container.config.from_yaml(config_file)
    
    # Override with environment variables
    container.config.from_env("HR_MATCHER", separator="__")
    
    return container


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = create_container()
    return _container


def set_container(container: Container) -> None:
    """Set the global container instance (useful for testing)."""
    global _container
    _container = container


def reset_container() -> None:
    """Reset the global container instance."""
    global _container
    if _container:
        _container.shutdown_resources()
    _container = None


# Convenience functions for common dependencies
def get_logger():
    """Get logger instance."""
    return get_container().logger()


def get_metrics():
    """Get metrics instance."""
    return get_container().metrics()


def get_cache():
    """Get cache instance."""
    return get_container().cache()


def get_event_bus():
    """Get event bus instance."""
    return get_container().event_bus()
"""
Application configuration and settings management.

This module provides a comprehensive configuration system using Pydantic
for validation, environment variable support, and multiple configuration sources.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseSettings, Field, validator, SecretStr
from pydantic.networks import HttpUrl, PostgresDsn, RedisDsn

from src.shared.types import LogLevel


class Environment(str, Enum):
    """Application environments."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    # PostgreSQL settings
    postgres_host: str = Field("localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")
    postgres_user: str = Field("postgres", env="POSTGRES_USER")
    postgres_password: SecretStr = Field(..., env="POSTGRES_PASSWORD")
    postgres_database: str = Field("hr_matcher", env="POSTGRES_DATABASE")
    
    # Connection pool settings
    pool_size: int = Field(20, env="DB_POOL_SIZE")
    max_overflow: int = Field(10, env="DB_MAX_OVERFLOW")
    pool_timeout: float = Field(30.0, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(3600, env="DB_POOL_RECYCLE")
    
    # Event store settings
    event_store_connection: Optional[str] = Field(None, env="EVENT_STORE_CONNECTION")
    
    @property
    def postgres_dsn(self) -> PostgresDsn:
        """Build PostgreSQL DSN."""
        return PostgresDsn.build(
            scheme="postgresql",
            user=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=str(self.postgres_port),
            path=f"/{self.postgres_database}",
        )
    
    @property
    def async_postgres_dsn(self) -> str:
        """Build async PostgreSQL DSN."""
        return str(self.postgres_dsn).replace("postgresql://", "postgresql+asyncpg://")
    
    class Config:
        env_prefix = "DB_"


class CacheSettings(BaseSettings):
    """Cache configuration."""
    
    provider: str = Field("redis", env="CACHE_PROVIDER")  # redis or memory
    
    # Redis settings
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_db: int = Field(0, env="REDIS_DB")
    redis_password: Optional[SecretStr] = Field(None, env="REDIS_PASSWORD")
    redis_ssl: bool = Field(False, env="REDIS_SSL")
    
    # Cache behavior
    default_ttl: int = Field(3600, env="CACHE_DEFAULT_TTL")  # 1 hour
    key_prefix: str = Field("hr_matcher", env="CACHE_KEY_PREFIX")
    
    # Memory cache settings
    memory_max_size: int = Field(1000, env="CACHE_MEMORY_MAX_SIZE")
    
    @property
    def redis_dsn(self) -> Optional[RedisDsn]:
        """Build Redis DSN."""
        if self.provider != "redis":
            return None
        
        password = self.redis_password.get_secret_value() if self.redis_password else None
        return RedisDsn.build(
            scheme="rediss" if self.redis_ssl else "redis",
            user=None,
            password=password,
            host=self.redis_host,
            port=str(self.redis_port),
            path=f"/{self.redis_db}",
        )
    
    class Config:
        env_prefix = "CACHE_"


class MessagingSettings(BaseSettings):
    """Message queue configuration."""
    
    provider: str = Field("kafka", env="MESSAGING_PROVIDER")  # kafka or memory
    
    # Kafka settings
    kafka_bootstrap_servers: str = Field("localhost:9092", env="KAFKA_BOOTSTRAP_SERVERS")
    kafka_security_protocol: str = Field("PLAINTEXT", env="KAFKA_SECURITY_PROTOCOL")
    kafka_sasl_mechanism: Optional[str] = Field(None, env="KAFKA_SASL_MECHANISM")
    kafka_sasl_username: Optional[str] = Field(None, env="KAFKA_SASL_USERNAME")
    kafka_sasl_password: Optional[SecretStr] = Field(None, env="KAFKA_SASL_PASSWORD")
    
    # Topics
    events_topic: str = Field("hr_matcher.events", env="EVENTS_TOPIC")
    commands_topic: str = Field("hr_matcher.commands", env="COMMANDS_TOPIC")
    
    class Config:
        env_prefix = "MESSAGING_"


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""
    
    provider: str = Field("chroma", env="VECTOR_STORE_PROVIDER")
    persist_directory: str = Field("./data/chroma", env="VECTOR_STORE_PERSIST_DIR")
    collection_name: str = Field("hr_matcher", env="VECTOR_STORE_COLLECTION")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Performance settings
    batch_size: int = Field(100, env="VECTOR_STORE_BATCH_SIZE")
    n_results: int = Field(10, env="VECTOR_STORE_N_RESULTS")
    
    class Config:
        env_prefix = "VECTOR_"


class LLMSettings(BaseSettings):
    """LLM configuration."""
    
    provider: str = Field("ollama", env="LLM_PROVIDER")  # ollama, openai, anthropic
    
    # Ollama settings
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama2", env="OLLAMA_MODEL")
    
    # OpenAI settings
    openai_api_key: Optional[SecretStr] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(None, env="OPENAI_BASE_URL")
    
    # Common settings
    temperature: float = Field(0.7, env="LLM_TEMPERATURE")
    max_tokens: int = Field(2000, env="LLM_MAX_TOKENS")
    timeout: int = Field(30, env="LLM_TIMEOUT")
    max_retries: int = Field(3, env="LLM_MAX_RETRIES")
    
    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    class Config:
        env_prefix = "LLM_"


class APISettings(BaseSettings):
    """API configuration."""
    
    host: str = Field("0.0.0.0", env="API_HOST")
    port: int = Field(8000, env="API_PORT")
    workers: int = Field(4, env="API_WORKERS")
    
    # CORS settings
    cors_origins: List[str] = Field(["*"], env="CORS_ORIGINS")
    cors_credentials: bool = Field(True, env="CORS_CREDENTIALS")
    cors_methods: List[str] = Field(["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(["*"], env="CORS_HEADERS")
    
    # API documentation
    docs_url: Optional[str] = Field("/docs", env="API_DOCS_URL")
    redoc_url: Optional[str] = Field("/redoc", env="API_REDOC_URL")
    openapi_url: Optional[str] = Field("/openapi.json", env="API_OPENAPI_URL")
    
    # Security
    api_key_header: str = Field("X-API-Key", env="API_KEY_HEADER")
    api_keys: List[SecretStr] = Field([], env="API_KEYS")
    
    class Config:
        env_prefix = "API_"


class ResilienceSettings(BaseSettings):
    """Resilience and fault tolerance configuration."""
    
    # Circuit breaker
    circuit_breaker_threshold: int = Field(5, env="CIRCUIT_BREAKER_THRESHOLD")
    circuit_breaker_timeout: float = Field(60.0, env="CIRCUIT_BREAKER_TIMEOUT")
    circuit_breaker_success_threshold: int = Field(2, env="CIRCUIT_BREAKER_SUCCESS_THRESHOLD")
    
    # Rate limiting
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(60, env="RATE_LIMIT_WINDOW")  # seconds
    rate_limit_burst: int = Field(20, env="RATE_LIMIT_BURST")
    
    # Retry policy
    retry_max_attempts: int = Field(3, env="RETRY_MAX_ATTEMPTS")
    retry_delay: float = Field(1.0, env="RETRY_DELAY")
    retry_backoff_factor: float = Field(2.0, env="RETRY_BACKOFF_FACTOR")
    
    # Timeouts
    default_timeout: float = Field(30.0, env="DEFAULT_TIMEOUT")
    long_timeout: float = Field(300.0, env="LONG_TIMEOUT")
    
    class Config:
        env_prefix = "RESILIENCE_"


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    format: str = Field("json", env="LOG_FORMAT")  # json or text
    output_file: Optional[str] = Field(None, env="LOG_FILE")
    
    # Structured logging
    include_timestamp: bool = Field(True, env="LOG_TIMESTAMP")
    include_hostname: bool = Field(True, env="LOG_HOSTNAME")
    include_process: bool = Field(True, env="LOG_PROCESS")
    
    # Log rotation
    max_bytes: int = Field(10485760, env="LOG_MAX_BYTES")  # 10MB
    backup_count: int = Field(5, env="LOG_BACKUP_COUNT")
    
    class Config:
        env_prefix = "LOG_"


class MetricsSettings(BaseSettings):
    """Metrics and monitoring configuration."""
    
    enabled: bool = Field(True, env="METRICS_ENABLED")
    provider: str = Field("prometheus", env="METRICS_PROVIDER")  # prometheus or console
    
    # Prometheus settings
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    prometheus_path: str = Field("/metrics", env="PROMETHEUS_PATH")
    
    # Metrics behavior
    include_method_label: bool = Field(True, env="METRICS_INCLUDE_METHOD")
    include_status_label: bool = Field(True, env="METRICS_INCLUDE_STATUS")
    include_path_label: bool = Field(True, env="METRICS_INCLUDE_PATH")
    
    class Config:
        env_prefix = "METRICS_"


class ExternalAPISettings(BaseSettings):
    """External API configuration."""
    
    # LinkedIn settings
    linkedin_client_id: Optional[str] = Field(None, env="LINKEDIN_CLIENT_ID")
    linkedin_client_secret: Optional[SecretStr] = Field(None, env="LINKEDIN_CLIENT_SECRET")
    linkedin_redirect_uri: Optional[str] = Field(None, env="LINKEDIN_REDIRECT_URI")
    
    # Other APIs
    salary_api_key: Optional[SecretStr] = Field(None, env="SALARY_API_KEY")
    salary_api_base_url: str = Field("https://api.salary.com", env="SALARY_API_BASE_URL")
    
    # Timeouts
    external_api_timeout: float = Field(30.0, env="EXTERNAL_API_TIMEOUT")
    
    class Config:
        env_prefix = "EXTERNAL_"


class MatchingSettings(BaseSettings):
    """Matching algorithm configuration."""
    
    # Score weights
    skill_weight: float = Field(0.40, env="MATCH_SKILL_WEIGHT")
    experience_weight: float = Field(0.30, env="MATCH_EXPERIENCE_WEIGHT")
    education_weight: float = Field(0.20, env="MATCH_EDUCATION_WEIGHT")
    location_weight: float = Field(0.10, env="MATCH_LOCATION_WEIGHT")
    
    # Thresholds
    min_match_score: float = Field(0.5, env="MIN_MATCH_SCORE")
    high_match_threshold: float = Field(0.8, env="HIGH_MATCH_THRESHOLD")
    
    # Behavior
    use_semantic_matching: bool = Field(True, env="USE_SEMANTIC_MATCHING")
    max_matches_per_search: int = Field(100, env="MAX_MATCHES_PER_SEARCH")
    
    @validator("skill_weight", "experience_weight", "education_weight", "location_weight")
    def validate_weight(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Weight must be between 0 and 1")
        return v
    
    class Config:
        env_prefix = "MATCH_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application info
    app_name: str = Field("HR Matcher", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    environment: Environment = Field(Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # Feature flags
    enable_mcp_servers: bool = Field(True, env="ENABLE_MCP_SERVERS")
    enable_linkedin_integration: bool = Field(True, env="ENABLE_LINKEDIN")
    enable_salary_research: bool = Field(True, env="ENABLE_SALARY_RESEARCH")
    enable_async_processing: bool = Field(True, env="ENABLE_ASYNC")
    
    # Sub-configurations
    database: DatabaseSettings = DatabaseSettings()
    cache: CacheSettings = CacheSettings()
    messaging: MessagingSettings = MessagingSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    llm: LLMSettings = LLMSettings()
    api: APISettings = APISettings()
    resilience: ResilienceSettings = ResilienceSettings()
    logging: LoggingSettings = LoggingSettings()
    metrics: MetricsSettings = MetricsSettings()
    external_apis: ExternalAPISettings = ExternalAPISettings()
    matching: MatchingSettings = MatchingSettings()
    
    # Paths
    data_dir: Path = Field(Path("./data"), env="DATA_DIR")
    upload_dir: Path = Field(Path("./uploads"), env="UPLOAD_DIR")
    temp_dir: Path = Field(Path("./temp"), env="TEMP_DIR")
    
    @validator("data_dir", "upload_dir", "temp_dir")
    def create_directories(cls, v: Path) -> Path:
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    def get_dsn(self, service: str) -> Optional[str]:
        """Get DSN for a service."""
        if service == "postgres":
            return str(self.database.postgres_dsn)
        elif service == "redis":
            return str(self.cache.redis_dsn) if self.cache.redis_dsn else None
        return None
    
    def validate_weights(self) -> None:
        """Validate that matching weights sum to 1."""
        total = (
            self.matching.skill_weight +
            self.matching.experience_weight +
            self.matching.education_weight +
            self.matching.location_weight
        )
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Matching weights must sum to 1.0, got {total}")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Allow extra fields for forward compatibility
        extra = "allow"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton)."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.validate_weights()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None


# Configuration file support
def load_from_yaml(file_path: Union[str, Path]) -> Settings:
    """Load settings from YAML file."""
    import yaml
    
    with open(file_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    return Settings(**config_data)


def load_from_json(file_path: Union[str, Path]) -> Settings:
    """Load settings from JSON file."""
    import json
    
    with open(file_path, "r") as f:
        config_data = json.load(f)
    
    return Settings(**config_data)


def save_to_yaml(settings: Settings, file_path: Union[str, Path]) -> None:
    """Save settings to YAML file."""
    import yaml
    
    with open(file_path, "w") as f:
        yaml.dump(settings.dict(), f, default_flow_style=False)


def save_to_json(settings: Settings, file_path: Union[str, Path]) -> None:
    """Save settings to JSON file."""
    import json
    
    with open(file_path, "w") as f:
        json.dump(settings.dict(), f, indent=2)
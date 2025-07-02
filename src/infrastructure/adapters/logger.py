"""
Structured logging adapter implementation.

Provides a comprehensive logging solution with support for structured data,
context propagation, and multiple output formats.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pythonjsonlogger import jsonlogger

from src.shared.protocols import Logger


# Context variable for request/correlation tracking
correlation_id_var: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add severity
        log_record["severity"] = record.levelname
        
        # Add correlation ID if present
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_record["correlation_id"] = correlation_id
        
        # Add source location
        log_record["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Extract error details if present
        if record.exc_info:
            log_record["error"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "stacktrace": traceback.format_exception(*record.exc_info),
            }


class StructuredLogger(Logger):
    """Structured logger implementation."""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        json_format: bool = True,
        output_file: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper()))
        self._context = context or {}
        
        # Remove existing handlers to avoid duplicates
        self._logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if json_format:
            formatter = StructuredFormatter(
                "%(timestamp)s %(severity)s %(name)s %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        
        # File handler if specified
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
    
    def _merge_context(self, **kwargs: Any) -> Dict[str, Any]:
        """Merge logger context with provided kwargs."""
        merged = self._context.copy()
        merged.update(kwargs)
        
        # Convert special types
        for key, value in merged.items():
            if isinstance(value, UUID):
                merged[key] = str(value)
            elif hasattr(value, "__dict__"):
                # Convert objects to dict representation
                try:
                    merged[key] = vars(value)
                except:
                    merged[key] = str(value)
        
        return merged
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        extra = {"extra": self._merge_context(**kwargs)}
        self._logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        extra = {"extra": self._merge_context(**kwargs)}
        self._logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        extra = {"extra": self._merge_context(**kwargs)}
        self._logger.warning(message, extra=extra)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs: Any) -> None:
        """Log error message."""
        extra = {"extra": self._merge_context(**kwargs)}
        
        if error:
            extra["extra"]["error_type"] = type(error).__name__
            extra["extra"]["error_message"] = str(error)
            self._logger.error(message, exc_info=error, extra=extra)
        else:
            self._logger.error(message, extra=extra)
    
    def with_context(self, **context: Any) -> StructuredLogger:
        """Create logger with additional context."""
        merged_context = self._context.copy()
        merged_context.update(context)
        
        return StructuredLogger(
            name=self.name,
            level=self._logger.level,
            json_format=isinstance(
                self._logger.handlers[0].formatter, StructuredFormatter
            ),
            context=merged_context,
        )
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current context."""
        correlation_id_var.set(correlation_id)
    
    def clear_correlation_id(self) -> None:
        """Clear correlation ID."""
        correlation_id_var.set(None)
    
    def measure_duration(self, operation: str) -> DurationLogger:
        """Create a context manager for measuring operation duration."""
        return DurationLogger(self, operation)
    
    def log_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Log a structured event."""
        self.info(
            f"Event: {event_type}",
            event_type=event_type,
            event_data=event_data,
            **kwargs
        )
    
    def log_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        unit: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs: Any
    ) -> None:
        """Log a metric value."""
        metric_data = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "tags": tags or {},
        }
        
        self.info(
            f"Metric: {metric_name}={value}{unit or ''}",
            metric=metric_data,
            **kwargs
        )
    
    def log_audit(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        user_id: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        """Log an audit event."""
        audit_data = {
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "user_id": user_id,
            "changes": changes or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        self.info(
            f"Audit: {action} {entity_type} {entity_id}",
            audit=audit_data,
            **kwargs
        )


class DurationLogger:
    """Context manager for logging operation duration."""
    
    def __init__(self, logger: StructuredLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time: Optional[datetime] = None
        self.extra_context: Dict[str, Any] = {}
    
    def __enter__(self) -> DurationLogger:
        """Start timing."""
        self.start_time = datetime.now(timezone.utc)
        self.logger.debug(
            f"Starting {self.operation}",
            operation=self.operation,
            start_time=self.start_time.isoformat(),
        )
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End timing and log duration."""
        if not self.start_time:
            return
        
        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - self.start_time).total_seconds() * 1000
        
        log_data = {
            "operation": self.operation,
            "duration_ms": duration_ms,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            **self.extra_context,
        }
        
        if exc_type:
            log_data["error"] = {
                "type": exc_type.__name__,
                "message": str(exc_val),
            }
            self.logger.error(
                f"Failed {self.operation} after {duration_ms:.2f}ms",
                **log_data
            )
        else:
            self.logger.info(
                f"Completed {self.operation} in {duration_ms:.2f}ms",
                **log_data
            )
    
    def add_context(self, **kwargs: Any) -> None:
        """Add extra context to be logged."""
        self.extra_context.update(kwargs)


class LoggerFactory:
    """Factory for creating loggers with consistent configuration."""
    
    _default_config = {
        "level": "INFO",
        "json_format": True,
        "output_file": None,
    }
    
    _loggers: Dict[str, StructuredLogger] = {}
    
    @classmethod
    def configure_defaults(
        cls,
        level: str = "INFO",
        json_format: bool = True,
        output_file: Optional[str] = None,
    ) -> None:
        """Configure default logger settings."""
        cls._default_config.update({
            "level": level,
            "json_format": json_format,
            "output_file": output_file,
        })
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        **kwargs: Any
    ) -> StructuredLogger:
        """Get or create logger with given name."""
        if name not in cls._loggers:
            config = cls._default_config.copy()
            config.update(kwargs)
            cls._loggers[name] = StructuredLogger(name=name, **config)
        
        return cls._loggers[name]
    
    @classmethod
    def get_child_logger(
        cls,
        parent_name: str,
        child_name: str,
        **context: Any
    ) -> StructuredLogger:
        """Get child logger with inherited context."""
        full_name = f"{parent_name}.{child_name}"
        parent = cls.get_logger(parent_name)
        
        if full_name not in cls._loggers:
            cls._loggers[full_name] = parent.with_context(
                component=child_name,
                **context
            )
        
        return cls._loggers[full_name]
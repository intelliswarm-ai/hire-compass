"""
Metrics collection adapter implementations.

Provides Prometheus and console-based metrics collection for monitoring
application performance and business metrics.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
    start_http_server,
)

from src.shared.protocols import Logger, MetricsCollector


class MetricType:
    """Metric type constants."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class PrometheusMetrics(MetricsCollector):
    """Prometheus-based metrics implementation."""
    
    def __init__(
        self,
        namespace: str = "hr_matcher",
        port: Optional[int] = 9090,
        registry: Optional[CollectorRegistry] = None,
        start_server: bool = True,
    ):
        self.namespace = namespace
        self.port = port
        self.registry = registry or CollectorRegistry()
        
        # Metric storage
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._summaries: Dict[str, Summary] = {}
        
        # Start HTTP server if requested
        if start_server and port:
            start_http_server(port, registry=self.registry)
        
        # Initialize default metrics
        self._init_default_metrics()
    
    def _init_default_metrics(self) -> None:
        """Initialize default application metrics."""
        # Request metrics
        self._get_or_create_counter(
            "requests_total",
            "Total number of requests",
            ["method", "endpoint", "status"]
        )
        
        self._get_or_create_histogram(
            "request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"]
        )
        
        # Business metrics
        self._get_or_create_counter(
            "resumes_processed_total",
            "Total number of resumes processed",
            ["status"]
        )
        
        self._get_or_create_counter(
            "matches_calculated_total",
            "Total number of matches calculated",
            ["confidence_level"]
        )
        
        self._get_or_create_histogram(
            "match_score_distribution",
            "Distribution of match scores",
            ["position_level"]
        )
        
        # System metrics
        self._get_or_create_gauge(
            "active_connections",
            "Number of active connections",
            ["service"]
        )
        
        self._get_or_create_gauge(
            "cache_size_bytes",
            "Cache size in bytes",
            ["cache_type"]
        )
    
    def _get_or_create_counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Counter:
        """Get or create counter metric."""
        if name not in self._counters:
            self._counters[name] = Counter(
                f"{self.namespace}_{name}",
                description,
                labels or [],
                registry=self.registry
            )
        return self._counters[name]
    
    def _get_or_create_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Gauge:
        """Get or create gauge metric."""
        if name not in self._gauges:
            self._gauges[name] = Gauge(
                f"{self.namespace}_{name}",
                description,
                labels or [],
                registry=self.registry
            )
        return self._gauges[name]
    
    def _get_or_create_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """Get or create histogram metric."""
        if name not in self._histograms:
            self._histograms[name] = Histogram(
                f"{self.namespace}_{name}",
                description,
                labels or [],
                buckets=buckets,
                registry=self.registry
            )
        return self._histograms[name]
    
    def _get_or_create_summary(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Summary:
        """Get or create summary metric."""
        if name not in self._summaries:
            self._summaries[name] = Summary(
                f"{self.namespace}_{name}",
                description,
                labels or [],
                registry=self.registry
            )
        return self._summaries[name]
    
    def increment(
        self,
        metric: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment counter metric."""
        tags = tags or {}
        counter = self._get_or_create_counter(
            metric,
            f"Counter for {metric}",
            list(tags.keys())
        )
        
        if tags:
            counter.labels(**tags).inc(value)
        else:
            counter.inc(value)
    
    def gauge(
        self,
        metric: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set gauge metric."""
        tags = tags or {}
        gauge = self._get_or_create_gauge(
            metric,
            f"Gauge for {metric}",
            list(tags.keys())
        )
        
        if tags:
            gauge.labels(**tags).set(value)
        else:
            gauge.set(value)
    
    def histogram(
        self,
        metric: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record histogram metric."""
        tags = tags or {}
        histogram = self._get_or_create_histogram(
            metric,
            f"Histogram for {metric}",
            list(tags.keys())
        )
        
        if tags:
            histogram.labels(**tags).observe(value)
        else:
            histogram.observe(value)
    
    def timing(
        self,
        metric: str,
        duration: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record timing metric."""
        # Use histogram for timing metrics
        self.histogram(f"{metric}_duration_seconds", duration, tags)
    
    @contextmanager
    def timer(
        self,
        metric: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.timing(metric, duration, tags)
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)
    
    def record_business_event(
        self,
        event_type: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record business event with attributes."""
        attributes = attributes or {}
        
        # Convert attributes to tags
        tags = {
            k: str(v) for k, v in attributes.items()
            if isinstance(v, (str, int, float, bool))
        }
        
        # Record event counter
        self.increment(f"business_event_{event_type}_total", tags=tags)
        
        # Record specific metrics based on event type
        if event_type == "resume_parsed":
            if "duration_ms" in attributes:
                self.histogram(
                    "resume_parsing_duration_ms",
                    attributes["duration_ms"],
                    tags={"parser_type": attributes.get("parser_type", "default")}
                )
        
        elif event_type == "match_calculated":
            if "score" in attributes:
                self.histogram(
                    "match_score_distribution",
                    attributes["score"],
                    tags={"confidence": attributes.get("confidence", "medium")}
                )


class ConsoleMetrics(MetricsCollector):
    """Console-based metrics implementation for development."""
    
    def __init__(self, logger: Optional[Logger] = None, print_interval: float = 60.0):
        self.logger = logger
        self.print_interval = print_interval
        self.last_print_time = time.time()
        
        # Metric storage
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._histograms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._timings: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    def increment(
        self,
        metric: str,
        value: int = 1,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment counter metric."""
        tag_key = self._tag_key(tags)
        self._counters[metric][tag_key] += value
        
        if self.logger:
            self.logger.debug(
                f"Metric increment: {metric}={value}",
                metric_type="counter",
                tags=tags
            )
        
        self._maybe_print_summary()
    
    def gauge(
        self,
        metric: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Set gauge metric."""
        tag_key = self._tag_key(tags)
        self._gauges[metric][tag_key] = value
        
        if self.logger:
            self.logger.debug(
                f"Metric gauge: {metric}={value}",
                metric_type="gauge",
                tags=tags
            )
        
        self._maybe_print_summary()
    
    def histogram(
        self,
        metric: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record histogram metric."""
        tag_key = self._tag_key(tags)
        self._histograms[metric][tag_key].append(value)
        
        # Keep only last 1000 values per metric/tag combination
        if len(self._histograms[metric][tag_key]) > 1000:
            self._histograms[metric][tag_key] = self._histograms[metric][tag_key][-1000:]
        
        if self.logger:
            self.logger.debug(
                f"Metric histogram: {metric}={value}",
                metric_type="histogram",
                tags=tags
            )
        
        self._maybe_print_summary()
    
    def timing(
        self,
        metric: str,
        duration: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record timing metric."""
        tag_key = self._tag_key(tags)
        self._timings[metric][tag_key].append(duration)
        
        # Keep only last 1000 values
        if len(self._timings[metric][tag_key]) > 1000:
            self._timings[metric][tag_key] = self._timings[metric][tag_key][-1000:]
        
        if self.logger:
            self.logger.debug(
                f"Metric timing: {metric}={duration:.3f}s",
                metric_type="timing",
                tags=tags
            )
        
        self._maybe_print_summary()
    
    def _tag_key(self, tags: Optional[Dict[str, str]]) -> str:
        """Create key from tags."""
        if not tags:
            return "_"
        return "|".join(f"{k}={v}" for k, v in sorted(tags.items()))
    
    def _maybe_print_summary(self) -> None:
        """Print summary if interval has passed."""
        current_time = time.time()
        if current_time - self.last_print_time >= self.print_interval:
            self.print_summary()
            self.last_print_time = current_time
    
    def print_summary(self) -> None:
        """Print metrics summary."""
        print("\n" + "=" * 60)
        print(f"Metrics Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Print counters
        if self._counters:
            print("\nCounters:")
            for metric, values in sorted(self._counters.items()):
                for tags, value in sorted(values.items()):
                    print(f"  {metric}{{{tags}}}: {value:.0f}")
        
        # Print gauges
        if self._gauges:
            print("\nGauges:")
            for metric, values in sorted(self._gauges.items()):
                for tags, value in sorted(values.items()):
                    print(f"  {metric}{{{tags}}}: {value:.2f}")
        
        # Print histogram summaries
        if self._histograms:
            print("\nHistograms:")
            for metric, tag_values in sorted(self._histograms.items()):
                for tags, values in sorted(tag_values.items()):
                    if values:
                        stats = self._calculate_stats(values)
                        print(f"  {metric}{{{tags}}}:")
                        print(f"    count: {stats['count']}")
                        print(f"    mean: {stats['mean']:.2f}")
                        print(f"    min: {stats['min']:.2f}")
                        print(f"    max: {stats['max']:.2f}")
                        print(f"    p50: {stats['p50']:.2f}")
                        print(f"    p95: {stats['p95']:.2f}")
                        print(f"    p99: {stats['p99']:.2f}")
        
        # Print timing summaries
        if self._timings:
            print("\nTimings:")
            for metric, tag_values in sorted(self._timings.items()):
                for tags, values in sorted(tag_values.items()):
                    if values:
                        stats = self._calculate_stats(values)
                        print(f"  {metric}{{{tags}}} (seconds):")
                        print(f"    count: {stats['count']}")
                        print(f"    mean: {stats['mean']:.3f}")
                        print(f"    min: {stats['min']:.3f}")
                        print(f"    max: {stats['max']:.3f}")
                        print(f"    p50: {stats['p50']:.3f}")
                        print(f"    p95: {stats['p95']:.3f}")
                        print(f"    p99: {stats['p99']:.3f}")
        
        print("=" * 60 + "\n")
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            "count": count,
            "mean": sum(sorted_values) / count if count > 0 else 0,
            "min": sorted_values[0] if count > 0 else 0,
            "max": sorted_values[-1] if count > 0 else 0,
            "p50": self._percentile(sorted_values, 0.5),
            "p95": self._percentile(sorted_values, 0.95),
            "p99": self._percentile(sorted_values, 0.99),
        }
    
    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        
        k = (len(sorted_values) - 1) * p
        f = int(k)
        c = k - f
        
        if f + 1 < len(sorted_values):
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
        else:
            return sorted_values[f]
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._timings.clear()
        self.last_print_time = time.time()


class MetricsDecorator:
    """Decorator for automatic metrics collection."""
    
    def __init__(
        self,
        collector: MetricsCollector,
        metric_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        record_args: bool = False,
        record_result: bool = False,
    ):
        self.collector = collector
        self.metric_name = metric_name
        self.tags = tags or {}
        self.record_args = record_args
        self.record_result = record_result
    
    def __call__(self, func: Callable) -> Callable:
        """Decorate function with metrics collection."""
        metric_name = self.metric_name or f"{func.__module__}.{func.__name__}"
        
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tags = self.tags.copy()
            
            # Add argument tags if requested
            if self.record_args and args:
                tags["first_arg"] = str(args[0])[:50]  # Limit length
            
            # Record call count
            self.collector.increment(f"{metric_name}_calls_total", tags=tags)
            
            # Time execution
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                
                # Record success
                tags["status"] = "success"
                self.collector.increment(f"{metric_name}_status_total", tags=tags)
                
                # Record result if requested
                if self.record_result and result is not None:
                    if isinstance(result, (int, float)):
                        self.collector.histogram(f"{metric_name}_result", result, tags=tags)
                
                return result
                
            except Exception as e:
                # Record failure
                tags["status"] = "failure"
                tags["error_type"] = type(e).__name__
                self.collector.increment(f"{metric_name}_status_total", tags=tags)
                raise
                
            finally:
                # Record duration
                duration = time.time() - start_time
                self.collector.timing(metric_name, duration, tags=tags)
        
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Similar implementation for sync functions
            tags = self.tags.copy()
            
            if self.record_args and args:
                tags["first_arg"] = str(args[0])[:50]
            
            self.collector.increment(f"{metric_name}_calls_total", tags=tags)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                tags["status"] = "success"
                self.collector.increment(f"{metric_name}_status_total", tags=tags)
                
                if self.record_result and result is not None:
                    if isinstance(result, (int, float)):
                        self.collector.histogram(f"{metric_name}_result", result, tags=tags)
                
                return result
                
            except Exception as e:
                tags["status"] = "failure"
                tags["error_type"] = type(e).__name__
                self.collector.increment(f"{metric_name}_status_total", tags=tags)
                raise
                
            finally:
                duration = time.time() - start_time
                self.collector.timing(metric_name, duration, tags=tags)
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
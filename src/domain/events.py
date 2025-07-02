"""
Domain events for event-driven architecture.

Domain events capture things that happen in the domain that other parts
of the system might be interested in. They are immutable records of
something that has occurred.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from src.shared.types import (
    CandidateId,
    CompanyId,
    EventMetadata,
    EventPayload,
    MatchConfidence,
    PositionId,
    ResumeId,
    Score,
    UserId,
)


@dataclass(frozen=True)
class DomainEvent(ABC):
    """Base class for all domain events."""
    
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=datetime.now)
    user_id: Optional[UserId] = None
    correlation_id: Optional[UUID] = None
    causation_id: Optional[UUID] = None
    
    @property
    def event_type(self) -> str:
        """Get event type from class name."""
        return self.__class__.__name__
    
    @property
    def aggregate_id(self) -> UUID:
        """Get the aggregate ID this event relates to."""
        # This should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement aggregate_id property")
    
    @property
    def aggregate_type(self) -> str:
        """Get the aggregate type this event relates to."""
        # This should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement aggregate_type property")
    
    def to_metadata(self) -> EventMetadata:
        """Convert to event metadata."""
        return EventMetadata(
            event_id=str(self.event_id),
            event_type=self.event_type,
            aggregate_id=str(self.aggregate_id),
            aggregate_type=self.aggregate_type,
            occurred_at=self.occurred_at,
            user_id=str(self.user_id) if self.user_id else None,
            correlation_id=str(self.correlation_id) if self.correlation_id else None,
            causation_id=str(self.causation_id) if self.causation_id else None,
        )
    
    @property
    def payload(self) -> EventPayload:
        """Get event payload for serialization."""
        # Convert dataclass to dict, excluding base fields
        base_fields = {'event_id', 'occurred_at', 'user_id', 'correlation_id', 'causation_id'}
        return {
            k: v for k, v in self.__dict__.items()
            if k not in base_fields and not k.startswith('_')
        }


# Resume Events

@dataclass(frozen=True)
class ResumeCreated(DomainEvent):
    """Event raised when a resume is created."""
    
    resume_id: ResumeId
    candidate_id: CandidateId
    candidate_name: str
    
    @property
    def aggregate_id(self) -> UUID:
        return self.resume_id
    
    @property
    def aggregate_type(self) -> str:
        return "Resume"


@dataclass(frozen=True)
class ResumeUpdated(DomainEvent):
    """Event raised when a resume is updated."""
    
    resume_id: ResumeId
    fields_updated: list[str]
    
    @property
    def aggregate_id(self) -> UUID:
        return self.resume_id
    
    @property
    def aggregate_type(self) -> str:
        return "Resume"


@dataclass(frozen=True)
class ResumeDeleted(DomainEvent):
    """Event raised when a resume is deleted."""
    
    resume_id: ResumeId
    reason: Optional[str] = None
    
    @property
    def aggregate_id(self) -> UUID:
        return self.resume_id
    
    @property
    def aggregate_type(self) -> str:
        return "Resume"


@dataclass(frozen=True)
class ResumeParsed(DomainEvent):
    """Event raised when a resume file is successfully parsed."""
    
    resume_id: ResumeId
    file_path: str
    parsing_duration_ms: int
    extracted_skills_count: int
    extracted_experience_years: float
    
    @property
    def aggregate_id(self) -> UUID:
        return self.resume_id
    
    @property
    def aggregate_type(self) -> str:
        return "Resume"


@dataclass(frozen=True)
class ResumeSkillsUpdated(DomainEvent):
    """Event raised when resume skills are updated."""
    
    resume_id: ResumeId
    skills_added: list[str]
    skills_removed: list[str]
    total_skills: int
    
    @property
    def aggregate_id(self) -> UUID:
        return self.resume_id
    
    @property
    def aggregate_type(self) -> str:
        return "Resume"


@dataclass(frozen=True)
class ResumeExperienceAdded(DomainEvent):
    """Event raised when experience is added to resume."""
    
    resume_id: ResumeId
    experience_id: UUID
    company: str
    position: str
    duration_months: int
    is_current: bool
    
    @property
    def aggregate_id(self) -> UUID:
        return self.resume_id
    
    @property
    def aggregate_type(self) -> str:
        return "Resume"


# Position Events

@dataclass(frozen=True)
class PositionCreated(DomainEvent):
    """Event raised when a position is created."""
    
    position_id: PositionId
    company_id: CompanyId
    title: str
    location: str
    
    @property
    def aggregate_id(self) -> UUID:
        return self.position_id
    
    @property
    def aggregate_type(self) -> str:
        return "Position"


@dataclass(frozen=True)
class PositionUpdated(DomainEvent):
    """Event raised when a position is updated."""
    
    position_id: PositionId
    fields_updated: list[str]
    
    @property
    def aggregate_id(self) -> UUID:
        return self.position_id
    
    @property
    def aggregate_type(self) -> str:
        return "Position"


@dataclass(frozen=True)
class PositionActivated(DomainEvent):
    """Event raised when a position is activated."""
    
    position_id: PositionId
    
    @property
    def aggregate_id(self) -> UUID:
        return self.position_id
    
    @property
    def aggregate_type(self) -> str:
        return "Position"


@dataclass(frozen=True)
class PositionDeactivated(DomainEvent):
    """Event raised when a position is deactivated."""
    
    position_id: PositionId
    reason: Optional[str] = None
    
    @property
    def aggregate_id(self) -> UUID:
        return self.position_id
    
    @property
    def aggregate_type(self) -> str:
        return "Position"


@dataclass(frozen=True)
class PositionExpired(DomainEvent):
    """Event raised when a position expires."""
    
    position_id: PositionId
    expired_date: datetime
    days_open: int
    
    @property
    def aggregate_id(self) -> UUID:
        return self.position_id
    
    @property
    def aggregate_type(self) -> str:
        return "Position"


@dataclass(frozen=True)
class PositionViewsIncremented(DomainEvent):
    """Event raised when position view count increases."""
    
    position_id: PositionId
    new_view_count: int
    viewer_type: str  # "candidate", "internal", "anonymous"
    
    @property
    def aggregate_id(self) -> UUID:
        return self.position_id
    
    @property
    def aggregate_type(self) -> str:
        return "Position"


# Matching Events

@dataclass(frozen=True)
class MatchCalculated(DomainEvent):
    """Event raised when a match is calculated."""
    
    match_id: UUID
    resume_id: ResumeId
    position_id: PositionId
    overall_score: Score
    confidence: MatchConfidence
    
    @property
    def aggregate_id(self) -> UUID:
        return self.match_id
    
    @property
    def aggregate_type(self) -> str:
        return "Match"


@dataclass(frozen=True)
class BatchMatchingStarted(DomainEvent):
    """Event raised when batch matching starts."""
    
    batch_id: UUID
    resume_count: int
    position_count: int
    total_comparisons: int
    
    @property
    def aggregate_id(self) -> UUID:
        return self.batch_id
    
    @property
    def aggregate_type(self) -> str:
        return "BatchMatch"


@dataclass(frozen=True)
class BatchMatchingCompleted(DomainEvent):
    """Event raised when batch matching completes."""
    
    batch_id: UUID
    duration_seconds: float
    matches_found: int
    high_confidence_matches: int
    average_score: float
    
    @property
    def aggregate_id(self) -> UUID:
        return self.batch_id
    
    @property
    def aggregate_type(self) -> str:
        return "BatchMatch"


@dataclass(frozen=True)
class HighScoreMatchFound(DomainEvent):
    """Event raised when a high-scoring match is found."""
    
    match_id: UUID
    resume_id: ResumeId
    position_id: PositionId
    candidate_name: str
    position_title: str
    overall_score: Score
    
    @property
    def aggregate_id(self) -> UUID:
        return self.match_id
    
    @property
    def aggregate_type(self) -> str:
        return "Match"


# Search Events

@dataclass(frozen=True)
class ResumeSearchPerformed(DomainEvent):
    """Event raised when resume search is performed."""
    
    search_id: UUID
    query: str
    filters: Dict[str, Any]
    result_count: int
    duration_ms: int
    
    @property
    def aggregate_id(self) -> UUID:
        return self.search_id
    
    @property
    def aggregate_type(self) -> str:
        return "Search"


@dataclass(frozen=True)
class PositionSearchPerformed(DomainEvent):
    """Event raised when position search is performed."""
    
    search_id: UUID
    query: str
    filters: Dict[str, Any]
    result_count: int
    duration_ms: int
    
    @property
    def aggregate_id(self) -> UUID:
        return self.search_id
    
    @property
    def aggregate_type(self) -> str:
        return "Search"


# Salary Research Events

@dataclass(frozen=True)
class SalaryResearchRequested(DomainEvent):
    """Event raised when salary research is requested."""
    
    research_id: UUID
    position_title: str
    location: str
    experience_level: str
    
    @property
    def aggregate_id(self) -> UUID:
        return self.research_id
    
    @property
    def aggregate_type(self) -> str:
        return "SalaryResearch"


@dataclass(frozen=True)
class SalaryResearchCompleted(DomainEvent):
    """Event raised when salary research completes."""
    
    research_id: UUID
    market_min: float
    market_max: float
    market_average: float
    data_sources: list[str]
    confidence_score: float
    
    @property
    def aggregate_id(self) -> UUID:
        return self.research_id
    
    @property
    def aggregate_type(self) -> str:
        return "SalaryResearch"


# Career Aspiration Events

@dataclass(frozen=True)
class AspirationAnalysisCompleted(DomainEvent):
    """Event raised when aspiration analysis completes."""
    
    analysis_id: UUID
    resume_id: ResumeId
    career_trajectory: str
    growth_areas: list[str]
    recommended_positions: list[str]
    
    @property
    def aggregate_id(self) -> UUID:
        return self.analysis_id
    
    @property
    def aggregate_type(self) -> str:
        return "AspirationAnalysis"


# System Events

@dataclass(frozen=True)
class SystemHealthCheckPerformed(DomainEvent):
    """Event raised when system health check is performed."""
    
    check_id: UUID
    services_checked: list[str]
    healthy_services: list[str]
    unhealthy_services: list[str]
    overall_status: str
    
    @property
    def aggregate_id(self) -> UUID:
        return self.check_id
    
    @property
    def aggregate_type(self) -> str:
        return "System"


@dataclass(frozen=True)
class RateLimitExceeded(DomainEvent):
    """Event raised when rate limit is exceeded."""
    
    limit_id: UUID
    client_id: str
    endpoint: str
    limit_type: str
    current_usage: int
    limit_threshold: int
    
    @property
    def aggregate_id(self) -> UUID:
        return self.limit_id
    
    @property
    def aggregate_type(self) -> str:
        return "RateLimit"


@dataclass(frozen=True)
class CircuitBreakerOpened(DomainEvent):
    """Event raised when circuit breaker opens."""
    
    breaker_id: UUID
    service_name: str
    failure_count: int
    failure_threshold: int
    last_error: str
    
    @property
    def aggregate_id(self) -> UUID:
        return self.breaker_id
    
    @property
    def aggregate_type(self) -> str:
        return "CircuitBreaker"


@dataclass(frozen=True)
class CircuitBreakerClosed(DomainEvent):
    """Event raised when circuit breaker closes."""
    
    breaker_id: UUID
    service_name: str
    recovery_duration_seconds: float
    
    @property
    def aggregate_id(self) -> UUID:
        return self.breaker_id
    
    @property
    def aggregate_type(self) -> str:
        return "CircuitBreaker"
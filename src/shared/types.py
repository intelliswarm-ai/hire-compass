"""
Shared type definitions and domain primitives.

This module defines the core types used throughout the application,
ensuring type safety and domain modeling best practices.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import (
    Any,
    Dict,
    List,
    NewType,
    Optional,
    TypeVar,
    Union,
    Protocol,
    runtime_checkable,
    TypedDict,
    Final,
    Literal,
    TypeAlias,
)
from uuid import UUID
from enum import Enum, auto
import re

# Type variables for generics
T = TypeVar('T')
TEntity = TypeVar('TEntity', bound='Entity')
TValueObject = TypeVar('TValueObject', bound='ValueObject')
TAggregate = TypeVar('TAggregate', bound='AggregateRoot')

# Domain primitive types using NewType for type safety
ResumeId = NewType('ResumeId', UUID)
PositionId = NewType('PositionId', UUID)
CandidateId = NewType('CandidateId', UUID)
CompanyId = NewType('CompanyId', UUID)
UserId = NewType('UserId', UUID)

# Score types with validation
Score = NewType('Score', float)  # 0.0 to 1.0
MatchScore = NewType('MatchScore', float)
ConfidenceScore = NewType('ConfidenceScore', float)

# Money type for salary
Money = NewType('Money', Decimal)

# Type aliases for clarity
Email: TypeAlias = str
PhoneNumber: TypeAlias = str
URL: TypeAlias = str
ISO8601DateTime: TypeAlias = str
SkillName: TypeAlias = str
YearsOfExperience: TypeAlias = float


class DomainError(Exception):
    """Base exception for domain errors."""
    
    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code or self.__class__.__name__


class ValidationError(DomainError):
    """Raised when domain validation fails."""
    pass


class BusinessRuleViolation(DomainError):
    """Raised when a business rule is violated."""
    pass


class ExperienceLevel(str, Enum):
    """Experience level enumeration with ordering support."""
    
    INTERN = "intern"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    EXECUTIVE = "executive"
    
    @property
    def min_years(self) -> float:
        """Minimum years of experience for this level."""
        mapping = {
            self.INTERN: 0,
            self.ENTRY: 0,
            self.MID: 3,
            self.SENIOR: 5,
            self.LEAD: 8,
            self.PRINCIPAL: 10,
            self.EXECUTIVE: 12,
        }
        return mapping[self]
    
    @property
    def rank(self) -> int:
        """Numeric rank for comparison."""
        return list(self.__class__).index(self)
    
    def __lt__(self, other: ExperienceLevel) -> bool:
        return self.rank < other.rank


class EducationLevel(str, Enum):
    """Education level enumeration with ordering."""
    
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELORS = "bachelors"
    MASTERS = "masters"
    PHD = "phd"
    
    @property
    def rank(self) -> int:
        return list(self.__class__).index(self)
    
    def __lt__(self, other: EducationLevel) -> bool:
        return self.rank < other.rank


class WorkMode(str, Enum):
    """Work mode preferences."""
    
    ONSITE = "onsite"
    REMOTE = "remote"
    HYBRID = "hybrid"


class MatchConfidence(str, Enum):
    """Confidence level for matches."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PERFECT = "perfect"
    
    @classmethod
    def from_score(cls, score: float) -> MatchConfidence:
        """Determine confidence level from score."""
        if score >= 0.95:
            return cls.PERFECT
        elif score >= 0.80:
            return cls.HIGH
        elif score >= 0.60:
            return cls.MEDIUM
        else:
            return cls.LOW


class EmploymentType(str, Enum):
    """Employment type options."""
    
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    FREELANCE = "freelance"


# Validation utilities
class Validators:
    """Domain validation utilities."""
    
    EMAIL_REGEX: Final = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    PHONE_REGEX: Final = re.compile(
        r'^[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}$'
    )
    
    URL_REGEX: Final = re.compile(
        r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)$'
    )
    
    @staticmethod
    def validate_email(email: str) -> Email:
        """Validate and return email address."""
        if not email or not Validators.EMAIL_REGEX.match(email):
            raise ValidationError(f"Invalid email address: {email}")
        return email.lower()
    
    @staticmethod
    def validate_phone(phone: str) -> PhoneNumber:
        """Validate and return phone number."""
        if not phone or not Validators.PHONE_REGEX.match(phone):
            raise ValidationError(f"Invalid phone number: {phone}")
        return phone
    
    @staticmethod
    def validate_url(url: str) -> URL:
        """Validate and return URL."""
        if not url or not Validators.URL_REGEX.match(url):
            raise ValidationError(f"Invalid URL: {url}")
        return url
    
    @staticmethod
    def validate_score(score: float, name: str = "score") -> Score:
        """Validate score is between 0 and 1."""
        if not 0 <= score <= 1:
            raise ValidationError(f"{name} must be between 0 and 1, got {score}")
        return Score(score)
    
    @staticmethod
    def validate_money(amount: Union[int, float, Decimal]) -> Money:
        """Validate and return money amount."""
        decimal_amount = Decimal(str(amount))
        if decimal_amount < 0:
            raise ValidationError(f"Money amount cannot be negative: {amount}")
        return Money(decimal_amount.quantize(Decimal('0.01')))
    
    @staticmethod
    def validate_years_of_experience(years: float) -> YearsOfExperience:
        """Validate years of experience."""
        if years < 0:
            raise ValidationError(f"Years of experience cannot be negative: {years}")
        if years > 100:
            raise ValidationError(f"Years of experience unrealistic: {years}")
        return YearsOfExperience(years)


# Result type for operations that can fail
@runtime_checkable
class Result(Protocol[T]):
    """Result type for operations that can fail."""
    
    @property
    def is_success(self) -> bool:
        """Check if the result is successful."""
        ...
    
    @property
    def is_failure(self) -> bool:
        """Check if the result is a failure."""
        ...
    
    @property
    def value(self) -> Optional[T]:
        """Get the success value if available."""
        ...
    
    @property
    def error(self) -> Optional[Exception]:
        """Get the error if available."""
        ...


class Success(Result[T]):
    """Successful result."""
    
    def __init__(self, value: T) -> None:
        self._value = value
    
    @property
    def is_success(self) -> bool:
        return True
    
    @property
    def is_failure(self) -> bool:
        return False
    
    @property
    def value(self) -> Optional[T]:
        return self._value
    
    @property
    def error(self) -> Optional[Exception]:
        return None


class Failure(Result[T]):
    """Failed result."""
    
    def __init__(self, error: Exception) -> None:
        self._error = error
    
    @property
    def is_success(self) -> bool:
        return False
    
    @property
    def is_failure(self) -> bool:
        return True
    
    @property
    def value(self) -> Optional[T]:
        return None
    
    @property
    def error(self) -> Optional[Exception]:
        return self._error


# Event types
EventType: TypeAlias = str
EventPayload: TypeAlias = Dict[str, Any]


class EventMetadata(TypedDict):
    """Metadata for domain events."""
    
    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    occurred_at: datetime
    user_id: Optional[str]
    correlation_id: Optional[str]
    causation_id: Optional[str]


# Command and Query types
CommandType: TypeAlias = str
QueryType: TypeAlias = str


class CommandMetadata(TypedDict):
    """Metadata for commands."""
    
    command_id: str
    command_type: str
    user_id: Optional[str]
    correlation_id: Optional[str]
    timestamp: datetime


class QueryMetadata(TypedDict):
    """Metadata for queries."""
    
    query_id: str
    query_type: str
    user_id: Optional[str]
    correlation_id: Optional[str]
    timestamp: datetime


# Pagination types
class PaginationParams(TypedDict):
    """Pagination parameters."""
    
    page: int
    page_size: int
    sort_by: Optional[str]
    sort_order: Literal["asc", "desc"]


class PaginatedResult(TypedDict, Generic[T]):
    """Paginated result container."""
    
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int


# Filter types
FilterOperator: TypeAlias = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "like", "contains"]


class FilterCondition(TypedDict):
    """Single filter condition."""
    
    field: str
    operator: FilterOperator
    value: Any


class FilterExpression(TypedDict):
    """Complex filter expression."""
    
    conditions: List[FilterCondition]
    logic: Literal["and", "or"]


# Time range types
class TimeRange(TypedDict):
    """Time range for queries."""
    
    start: datetime
    end: datetime


# Audit types
class AuditInfo(TypedDict):
    """Audit information for entities."""
    
    created_at: datetime
    created_by: Optional[UserId]
    updated_at: Optional[datetime]
    updated_by: Optional[UserId]
    version: int
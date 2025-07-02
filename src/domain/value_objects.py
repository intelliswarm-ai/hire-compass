"""
Domain value objects implementing immutability and validation.

Value objects are immutable objects that represent descriptive aspects of the domain
with no conceptual identity. They are instantiated to represent elements of the design
that we care about only for what they are, not who or which they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

from src.shared.types import (
    ConfidenceScore,
    EducationLevel,
    Email,
    ExperienceLevel,
    MatchConfidence,
    Money,
    PhoneNumber,
    Score,
    SkillName,
    URL,
    ValidationError,
    Validators,
    WorkMode,
    YearsOfExperience,
)


@dataclass(frozen=True)
class ValueObject:
    """Base class for value objects."""
    
    def __post_init__(self) -> None:
        """Validate after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Override to implement validation logic."""
        pass
    
    def __eq__(self, other: Any) -> bool:
        """Value objects are equal if all attributes are equal."""
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__
    
    def __hash__(self) -> int:
        """Hash based on all attributes."""
        return hash(tuple(sorted(self.__dict__.items())))


@dataclass(frozen=True)
class EmailAddress(ValueObject):
    """Email address value object."""
    
    value: Email
    
    def __post_init__(self) -> None:
        """Validate email on creation."""
        object.__setattr__(self, 'value', Validators.validate_email(self.value))
        super().__post_init__()
    
    @classmethod
    def create(cls, email: str) -> EmailAddress:
        """Factory method to create email address."""
        return cls(value=email)
    
    def domain(self) -> str:
        """Get email domain."""
        return self.value.split('@')[1]
    
    def is_corporate(self) -> bool:
        """Check if email is from a corporate domain."""
        free_domains = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'}
        return self.domain() not in free_domains
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Phone(ValueObject):
    """Phone number value object."""
    
    value: PhoneNumber
    country_code: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate phone on creation."""
        object.__setattr__(self, 'value', Validators.validate_phone(self.value))
        super().__post_init__()
    
    @classmethod
    def create(cls, phone: str, country_code: Optional[str] = None) -> Phone:
        """Factory method to create phone number."""
        return cls(value=phone, country_code=country_code)
    
    def formatted(self) -> str:
        """Get formatted phone number."""
        # Simple formatting, can be enhanced
        return self.value
    
    def __str__(self) -> str:
        return self.formatted()


@dataclass(frozen=True)
class Address(ValueObject):
    """Physical address value object."""
    
    street: str
    city: str
    state_province: str
    postal_code: str
    country: str
    
    def _validate(self) -> None:
        """Validate address components."""
        if not all([self.street, self.city, self.state_province, self.postal_code, self.country]):
            raise ValidationError("All address fields are required")
        
        if len(self.country) != 2:
            raise ValidationError("Country must be a 2-letter ISO code")
    
    def formatted(self, multiline: bool = False) -> str:
        """Get formatted address."""
        parts = [
            self.street,
            f"{self.city}, {self.state_province} {self.postal_code}",
            self.country
        ]
        separator = "\n" if multiline else ", "
        return separator.join(parts)
    
    def __str__(self) -> str:
        return self.formatted()


@dataclass(frozen=True)
class Location(ValueObject):
    """Geographic location value object."""
    
    city: str
    state_province: Optional[str] = None
    country: str = "US"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    def _validate(self) -> None:
        """Validate location data."""
        if not self.city:
            raise ValidationError("City is required")
        
        if self.latitude is not None:
            if not -90 <= self.latitude <= 90:
                raise ValidationError("Latitude must be between -90 and 90")
        
        if self.longitude is not None:
            if not -180 <= self.longitude <= 180:
                raise ValidationError("Longitude must be between -180 and 180")
    
    def formatted(self) -> str:
        """Get formatted location string."""
        parts = [self.city]
        if self.state_province:
            parts.append(self.state_province)
        parts.append(self.country)
        return ", ".join(parts)
    
    def has_coordinates(self) -> bool:
        """Check if location has geographic coordinates."""
        return self.latitude is not None and self.longitude is not None
    
    def distance_to(self, other: Location) -> Optional[float]:
        """Calculate distance to another location in kilometers."""
        if not (self.has_coordinates() and other.has_coordinates()):
            return None
        
        # Haversine formula for distance calculation
        import math
        
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def __str__(self) -> str:
        return self.formatted()


@dataclass(frozen=True)
class Skill(ValueObject):
    """Skill value object with proficiency and experience."""
    
    name: SkillName
    proficiency: Optional[str] = None
    years_of_experience: Optional[YearsOfExperience] = None
    last_used: Optional[datetime] = None
    
    PROFICIENCY_LEVELS: ClassVar[List[str]] = ["Beginner", "Intermediate", "Advanced", "Expert"]
    
    def _validate(self) -> None:
        """Validate skill data."""
        if not self.name or len(self.name) < 2:
            raise ValidationError("Skill name must be at least 2 characters")
        
        if self.proficiency and self.proficiency not in self.PROFICIENCY_LEVELS:
            raise ValidationError(f"Proficiency must be one of: {', '.join(self.PROFICIENCY_LEVELS)}")
        
        if self.years_of_experience is not None:
            Validators.validate_years_of_experience(self.years_of_experience)
        
        if self.last_used and self.last_used > datetime.now():
            raise ValidationError("Last used date cannot be in the future")
    
    def is_current(self, months_threshold: int = 24) -> bool:
        """Check if skill is current based on last used date."""
        if not self.last_used:
            return True  # Assume current if no date provided
        
        threshold_date = datetime.now() - timedelta(days=months_threshold * 30)
        return self.last_used >= threshold_date
    
    def proficiency_score(self) -> float:
        """Get numeric proficiency score (0-1)."""
        if not self.proficiency:
            return 0.5  # Default to intermediate
        
        proficiency_map = {
            "Beginner": 0.25,
            "Intermediate": 0.5,
            "Advanced": 0.75,
            "Expert": 1.0
        }
        return proficiency_map.get(self.proficiency, 0.5)
    
    def __str__(self) -> str:
        parts = [self.name]
        if self.proficiency:
            parts.append(f"({self.proficiency})")
        if self.years_of_experience:
            parts.append(f"{self.years_of_experience}y")
        return " ".join(parts)


@dataclass(frozen=True)
class SkillSet(ValueObject):
    """Collection of skills with categorization."""
    
    skills: Set[Skill] = field(default_factory=set)
    
    def add(self, skill: Skill) -> SkillSet:
        """Add a skill (returns new instance due to immutability)."""
        new_skills = self.skills.copy()
        new_skills.add(skill)
        return SkillSet(skills=new_skills)
    
    def remove(self, skill_name: str) -> SkillSet:
        """Remove a skill by name (returns new instance)."""
        new_skills = {s for s in self.skills if s.name != skill_name}
        return SkillSet(skills=new_skills)
    
    def get_by_name(self, name: str) -> Optional[Skill]:
        """Get skill by name."""
        return next((s for s in self.skills if s.name.lower() == name.lower()), None)
    
    def categorize(self) -> Dict[str, List[Skill]]:
        """Categorize skills by type."""
        categories = {
            "Programming Languages": ["Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "C#"],
            "Frameworks": ["React", "Angular", "Vue", "Django", "Flask", "Spring", "Express", "FastAPI"],
            "Databases": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra"],
            "Cloud": ["AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform"],
            "Tools": ["Git", "Jenkins", "JIRA", "Confluence", "Slack"],
            "Soft Skills": ["Leadership", "Communication", "Teamwork", "Problem Solving"]
        }
        
        categorized: Dict[str, List[Skill]] = {"Other": []}
        
        for skill in self.skills:
            categorized_flag = False
            for category, keywords in categories.items():
                if any(keyword.lower() in skill.name.lower() for keyword in keywords):
                    if category not in categorized:
                        categorized[category] = []
                    categorized[category].append(skill)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized["Other"].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categorized.items() if v}
    
    def match_percentage(self, required_skills: Set[str]) -> float:
        """Calculate percentage of required skills matched."""
        if not required_skills:
            return 100.0
        
        skill_names = {s.name.lower() for s in self.skills}
        required_lower = {s.lower() for s in required_skills}
        
        matched = len(skill_names.intersection(required_lower))
        return (matched / len(required_skills)) * 100
    
    def __len__(self) -> int:
        return len(self.skills)
    
    def __iter__(self):
        return iter(self.skills)


@dataclass(frozen=True)
class DateRange(ValueObject):
    """Date range value object."""
    
    start_date: datetime
    end_date: Optional[datetime] = None
    
    def _validate(self) -> None:
        """Validate date range."""
        if self.end_date and self.start_date > self.end_date:
            raise ValidationError("Start date cannot be after end date")
        
        if self.start_date > datetime.now():
            raise ValidationError("Start date cannot be in the future")
    
    @property
    def is_current(self) -> bool:
        """Check if date range is current (no end date)."""
        return self.end_date is None
    
    def duration_months(self) -> int:
        """Calculate duration in months."""
        end = self.end_date or datetime.now()
        months = (end.year - self.start_date.year) * 12
        months += end.month - self.start_date.month
        return max(0, months)
    
    def duration_years(self) -> float:
        """Calculate duration in years."""
        return round(self.duration_months() / 12, 1)
    
    def overlaps_with(self, other: DateRange) -> bool:
        """Check if date ranges overlap."""
        self_end = self.end_date or datetime.now()
        other_end = other.end_date or datetime.now()
        
        return not (self_end < other.start_date or other_end < self.start_date)
    
    def formatted(self) -> str:
        """Get formatted date range."""
        start = self.start_date.strftime("%B %Y")
        end = "Present" if self.is_current else self.end_date.strftime("%B %Y")
        return f"{start} - {end}"
    
    def __str__(self) -> str:
        return self.formatted()


@dataclass(frozen=True)
class SalaryRange(ValueObject):
    """Salary range value object."""
    
    minimum: Money
    maximum: Money
    currency: str = "USD"
    period: str = "yearly"
    
    def _validate(self) -> None:
        """Validate salary range."""
        if self.minimum > self.maximum:
            raise ValidationError("Minimum salary cannot be greater than maximum")
        
        if self.currency not in ["USD", "EUR", "GBP", "CAD", "AUD", "INR"]:
            raise ValidationError(f"Unsupported currency: {self.currency}")
        
        if self.period not in ["hourly", "daily", "weekly", "monthly", "yearly"]:
            raise ValidationError(f"Invalid salary period: {self.period}")
    
    def midpoint(self) -> Money:
        """Calculate midpoint of range."""
        return Money((self.minimum + self.maximum) / 2)
    
    def contains(self, amount: Money) -> bool:
        """Check if amount is within range."""
        return self.minimum <= amount <= self.maximum
    
    def overlaps_with(self, other: SalaryRange) -> bool:
        """Check if salary ranges overlap."""
        if self.currency != other.currency or self.period != other.period:
            return False  # Can't compare different currencies/periods
        
        return not (self.maximum < other.minimum or other.maximum < self.minimum)
    
    def to_yearly(self) -> SalaryRange:
        """Convert to yearly salary range."""
        if self.period == "yearly":
            return self
        
        multipliers = {
            "hourly": 2080,  # 40 hours/week * 52 weeks
            "daily": 260,    # 5 days/week * 52 weeks
            "weekly": 52,
            "monthly": 12
        }
        
        multiplier = multipliers.get(self.period, 1)
        return SalaryRange(
            minimum=Money(self.minimum * multiplier),
            maximum=Money(self.maximum * multiplier),
            currency=self.currency,
            period="yearly"
        )
    
    def formatted(self) -> str:
        """Get formatted salary range."""
        min_str = f"{self.currency} {self.minimum:,.0f}"
        max_str = f"{self.maximum:,.0f}"
        return f"{min_str} - {max_str} {self.period}"
    
    def __str__(self) -> str:
        return self.formatted()


@dataclass(frozen=True)
class MatchScoreDetails(ValueObject):
    """Detailed match score breakdown."""
    
    overall_score: Score
    skill_score: Score
    experience_score: Score
    education_score: Score
    location_score: Score
    salary_score: Score
    culture_score: Optional[Score] = None
    
    def _validate(self) -> None:
        """Validate all scores are between 0 and 1."""
        scores = [
            ("overall", self.overall_score),
            ("skill", self.skill_score),
            ("experience", self.experience_score),
            ("education", self.education_score),
            ("location", self.location_score),
            ("salary", self.salary_score)
        ]
        
        if self.culture_score is not None:
            scores.append(("culture", self.culture_score))
        
        for name, score in scores:
            Validators.validate_score(score, name)
    
    @property
    def confidence(self) -> MatchConfidence:
        """Get confidence level based on overall score."""
        return MatchConfidence.from_score(self.overall_score)
    
    def get_strengths(self, threshold: float = 0.8) -> List[str]:
        """Get areas where score exceeds threshold."""
        strengths = []
        
        if self.skill_score >= threshold:
            strengths.append("Strong skill match")
        if self.experience_score >= threshold:
            strengths.append("Ideal experience level")
        if self.education_score >= threshold:
            strengths.append("Education requirements met")
        if self.location_score >= threshold:
            strengths.append("Location compatible")
        if self.salary_score >= threshold:
            strengths.append("Salary expectations aligned")
        if self.culture_score and self.culture_score >= threshold:
            strengths.append("Cultural fit")
        
        return strengths
    
    def get_weaknesses(self, threshold: float = 0.6) -> List[str]:
        """Get areas where score is below threshold."""
        weaknesses = []
        
        if self.skill_score < threshold:
            weaknesses.append("Skills gap")
        if self.experience_score < threshold:
            weaknesses.append("Experience mismatch")
        if self.education_score < threshold:
            weaknesses.append("Education requirements not met")
        if self.location_score < threshold:
            weaknesses.append("Location challenges")
        if self.salary_score < threshold:
            weaknesses.append("Salary expectations misaligned")
        if self.culture_score and self.culture_score < threshold:
            weaknesses.append("Potential culture mismatch")
        
        return weaknesses
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        result = {
            "overall_score": float(self.overall_score),
            "skill_score": float(self.skill_score),
            "experience_score": float(self.experience_score),
            "education_score": float(self.education_score),
            "location_score": float(self.location_score),
            "salary_score": float(self.salary_score)
        }
        
        if self.culture_score is not None:
            result["culture_score"] = float(self.culture_score)
        
        return result
"""
Domain entities implementing business logic and invariants.

Entities are objects that have a distinct identity that runs through time
and different states. They implement the core business logic and maintain
consistency through invariants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Type
from uuid import UUID, uuid4

from src.shared.types import (
    AuditInfo,
    BusinessRuleViolation,
    CandidateId,
    CompanyId,
    EducationLevel,
    EmploymentType,
    ExperienceLevel,
    PositionId,
    ResumeId,
    UserId,
    ValidationError,
    WorkMode,
)
from src.domain.value_objects import (
    Address,
    DateRange,
    EmailAddress,
    Location,
    MatchScoreDetails,
    Phone,
    SalaryRange,
    Skill,
    SkillSet,
)
from src.domain.events import (
    DomainEvent,
    ResumeCreated,
    ResumeUpdated,
    PositionCreated,
    PositionUpdated,
    MatchCalculated,
)


@dataclass
class Entity(ABC):
    """Base class for all entities."""
    
    id: UUID
    audit_info: AuditInfo
    _events: List[DomainEvent] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize entity and validate invariants."""
        self._validate_invariants()
    
    @abstractmethod
    def _validate_invariants(self) -> None:
        """Validate business invariants. Must be called after any state change."""
        if not self.id:
            raise ValidationError("Entity ID is required")
    
    def add_event(self, event: DomainEvent) -> None:
        """Add domain event to be published."""
        self._events.append(event)
    
    def get_events(self) -> List[DomainEvent]:
        """Get pending domain events."""
        return self._events.copy()
    
    def clear_events(self) -> None:
        """Clear pending events after publishing."""
        self._events.clear()
    
    def mark_updated(self, user_id: Optional[UserId] = None) -> None:
        """Update audit info when entity is modified."""
        self.audit_info["updated_at"] = datetime.now()
        if user_id:
            self.audit_info["updated_by"] = user_id
        self.audit_info["version"] += 1
    
    def __eq__(self, other: Any) -> bool:
        """Entities are equal if they have the same ID."""
        if not isinstance(other, Entity):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash based on ID."""
        return hash(self.id)


@dataclass
class AggregateRoot(Entity):
    """Base class for aggregate roots."""
    
    def __init__(self, *args, **kwargs):
        """Initialize aggregate root."""
        super().__init__(*args, **kwargs)
        self._validate_invariants()
    
    def handle_command(self, command: Any) -> None:
        """Handle command and generate events."""
        # This would be implemented by specific aggregates
        pass


# Domain Entities

@dataclass
class Education:
    """Education entity."""
    
    id: UUID
    degree: str
    field_of_study: str
    institution: str
    location: Optional[Location]
    date_range: DateRange
    gpa: Optional[float]
    achievements: List[str]
    level: EducationLevel
    
    def __post_init__(self) -> None:
        """Validate education data."""
        if not self.degree or not self.institution:
            raise ValidationError("Degree and institution are required")
        
        if self.gpa is not None and not (0.0 <= self.gpa <= 4.0):
            raise ValidationError("GPA must be between 0.0 and 4.0")
    
    @classmethod
    def create(
        cls,
        degree: str,
        field_of_study: str,
        institution: str,
        date_range: DateRange,
        level: EducationLevel,
        location: Optional[Location] = None,
        gpa: Optional[float] = None,
        achievements: Optional[List[str]] = None,
    ) -> Education:
        """Factory method to create education."""
        return cls(
            id=uuid4(),
            degree=degree,
            field_of_study=field_of_study,
            institution=institution,
            location=location,
            date_range=date_range,
            gpa=gpa,
            achievements=achievements or [],
            level=level,
        )
    
    def is_completed(self) -> bool:
        """Check if education is completed."""
        return not self.date_range.is_current


@dataclass
class Experience:
    """Work experience entity."""
    
    id: UUID
    company: str
    company_id: Optional[CompanyId]
    position: str
    location: Optional[Location]
    date_range: DateRange
    description: str
    achievements: List[str]
    technologies: SkillSet
    employment_type: EmploymentType
    
    def __post_init__(self) -> None:
        """Validate experience data."""
        if not self.company or not self.position:
            raise ValidationError("Company and position are required")
    
    @classmethod
    def create(
        cls,
        company: str,
        position: str,
        date_range: DateRange,
        description: str,
        employment_type: EmploymentType = EmploymentType.FULL_TIME,
        company_id: Optional[CompanyId] = None,
        location: Optional[Location] = None,
        achievements: Optional[List[str]] = None,
        technologies: Optional[SkillSet] = None,
    ) -> Experience:
        """Factory method to create experience."""
        return cls(
            id=uuid4(),
            company=company,
            company_id=company_id,
            position=position,
            location=location,
            date_range=date_range,
            description=description,
            achievements=achievements or [],
            technologies=technologies or SkillSet(),
            employment_type=employment_type,
        )
    
    @property
    def is_current(self) -> bool:
        """Check if this is current employment."""
        return self.date_range.is_current
    
    def duration_months(self) -> int:
        """Get duration in months."""
        return self.date_range.duration_months()
    
    def infer_level(self) -> ExperienceLevel:
        """Infer experience level from position title."""
        title_lower = self.position.lower()
        
        if any(keyword in title_lower for keyword in ["intern", "trainee"]):
            return ExperienceLevel.INTERN
        elif any(keyword in title_lower for keyword in ["junior", "entry"]):
            return ExperienceLevel.ENTRY
        elif any(keyword in title_lower for keyword in ["senior", "sr."]):
            return ExperienceLevel.SENIOR
        elif any(keyword in title_lower for keyword in ["lead", "principal", "staff"]):
            return ExperienceLevel.LEAD
        elif any(keyword in title_lower for keyword in ["director", "vp", "vice president", "executive"]):
            return ExperienceLevel.EXECUTIVE
        else:
            return ExperienceLevel.MID


@dataclass
class Candidate:
    """Candidate entity."""
    
    id: CandidateId
    first_name: str
    last_name: str
    email: EmailAddress
    phone: Optional[Phone]
    location: Optional[Location]
    linkedin_url: Optional[str]
    github_url: Optional[str]
    portfolio_url: Optional[str]
    
    def __post_init__(self) -> None:
        """Validate candidate data."""
        if not self.first_name or not self.last_name:
            raise ValidationError("First and last name are required")
    
    @property
    def full_name(self) -> str:
        """Get full name."""
        return f"{self.first_name} {self.last_name}"
    
    @classmethod
    def create(
        cls,
        first_name: str,
        last_name: str,
        email: str,
        phone: Optional[str] = None,
        location: Optional[Location] = None,
        linkedin_url: Optional[str] = None,
        github_url: Optional[str] = None,
        portfolio_url: Optional[str] = None,
    ) -> Candidate:
        """Factory method to create candidate."""
        return cls(
            id=CandidateId(uuid4()),
            first_name=first_name,
            last_name=last_name,
            email=EmailAddress.create(email),
            phone=Phone.create(phone) if phone else None,
            location=location,
            linkedin_url=linkedin_url,
            github_url=github_url,
            portfolio_url=portfolio_url,
        )


@dataclass
class Resume(AggregateRoot):
    """Resume aggregate root."""
    
    candidate: Candidate
    professional_summary: Optional[str]
    education: List[Education]
    experience: List[Experience]
    skills: SkillSet
    certifications: List[str]
    languages: List[str]
    salary_expectation: Optional[SalaryRange]
    work_mode_preference: Optional[WorkMode]
    
    def _validate_invariants(self) -> None:
        """Validate resume invariants."""
        super()._validate_invariants()
        
        if not self.candidate:
            raise ValidationError("Resume must have a candidate")
        
        # Check for overlapping experiences
        for i, exp1 in enumerate(self.experience):
            for exp2 in self.experience[i+1:]:
                if exp1.date_range.overlaps_with(exp2.date_range):
                    if exp1.employment_type == EmploymentType.FULL_TIME and exp2.employment_type == EmploymentType.FULL_TIME:
                        raise BusinessRuleViolation("Cannot have overlapping full-time positions")
    
    @classmethod
    def create(
        cls,
        candidate: Candidate,
        created_by: Optional[UserId] = None,
        professional_summary: Optional[str] = None,
        education: Optional[List[Education]] = None,
        experience: Optional[List[Experience]] = None,
        skills: Optional[SkillSet] = None,
        certifications: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        salary_expectation: Optional[SalaryRange] = None,
        work_mode_preference: Optional[WorkMode] = None,
    ) -> Resume:
        """Factory method to create resume."""
        resume_id = ResumeId(uuid4())
        resume = cls(
            id=resume_id,
            audit_info=AuditInfo(
                created_at=datetime.now(),
                created_by=created_by,
                updated_at=None,
                updated_by=None,
                version=1,
            ),
            candidate=candidate,
            professional_summary=professional_summary,
            education=education or [],
            experience=experience or [],
            skills=skills or SkillSet(),
            certifications=certifications or [],
            languages=languages or [],
            salary_expectation=salary_expectation,
            work_mode_preference=work_mode_preference,
        )
        
        # Add creation event
        resume.add_event(ResumeCreated(
            resume_id=resume_id,
            candidate_id=candidate.id,
            candidate_name=candidate.full_name,
        ))
        
        return resume
    
    def update_skills(self, skills: SkillSet, updated_by: Optional[UserId] = None) -> None:
        """Update resume skills."""
        self.skills = skills
        self.mark_updated(updated_by)
        self._validate_invariants()
        
        self.add_event(ResumeUpdated(
            resume_id=ResumeId(self.id),
            fields_updated=["skills"],
        ))
    
    def add_experience(self, experience: Experience, updated_by: Optional[UserId] = None) -> None:
        """Add work experience."""
        self.experience.append(experience)
        # Sort by start date, most recent first
        self.experience.sort(key=lambda e: e.date_range.start_date, reverse=True)
        
        self.mark_updated(updated_by)
        self._validate_invariants()
        
        self.add_event(ResumeUpdated(
            resume_id=ResumeId(self.id),
            fields_updated=["experience"],
        ))
    
    def total_experience_years(self) -> float:
        """Calculate total years of experience."""
        if not self.experience:
            return 0.0
        
        # Account for overlapping experiences
        sorted_exp = sorted(self.experience, key=lambda e: e.date_range.start_date)
        
        total_months = 0
        current_end = None
        
        for exp in sorted_exp:
            start = exp.date_range.start_date
            end = exp.date_range.end_date or datetime.now()
            
            if current_end is None:
                # First experience
                total_months += exp.duration_months()
                current_end = end
            elif start > current_end:
                # No overlap
                total_months += exp.duration_months()
                current_end = end
            elif end > current_end:
                # Partial overlap
                overlap_months = (current_end.year - start.year) * 12 + (current_end.month - start.month)
                total_months += exp.duration_months() - overlap_months
                current_end = end
            # Else: fully contained within previous experience, don't add
        
        return round(total_months / 12, 1)
    
    def highest_education_level(self) -> Optional[EducationLevel]:
        """Get highest education level."""
        if not self.education:
            return None
        
        return max(edu.level for edu in self.education)
    
    def current_position(self) -> Optional[Experience]:
        """Get current position if any."""
        return next((exp for exp in self.experience if exp.is_current), None)
    
    def inferred_experience_level(self) -> ExperienceLevel:
        """Infer experience level from total experience and current position."""
        total_years = self.total_experience_years()
        current = self.current_position()
        
        # First check current position title
        if current:
            level = current.infer_level()
            if level != ExperienceLevel.MID:  # If we can infer from title, use it
                return level
        
        # Otherwise, use total years
        if total_years < 2:
            return ExperienceLevel.ENTRY
        elif total_years < 5:
            return ExperienceLevel.MID
        elif total_years < 8:
            return ExperienceLevel.SENIOR
        elif total_years < 12:
            return ExperienceLevel.LEAD
        else:
            return ExperienceLevel.EXECUTIVE


@dataclass
class Position(AggregateRoot):
    """Job position aggregate root."""
    
    title: str
    company_id: CompanyId
    department: str
    location: Location
    work_mode: WorkMode
    employment_type: EmploymentType
    description: str
    responsibilities: List[str]
    requirements: List[str]
    preferred_qualifications: List[str]
    required_skills: SkillSet
    preferred_skills: SkillSet
    experience_level: ExperienceLevel
    experience_range: Tuple[float, float]  # (min_years, max_years)
    education_requirements: List[EducationLevel]
    salary_range: Optional[SalaryRange]
    benefits: List[str]
    is_active: bool
    posted_date: datetime
    closing_date: Optional[datetime]
    
    def _validate_invariants(self) -> None:
        """Validate position invariants."""
        super()._validate_invariants()
        
        if not self.title or not self.description:
            raise ValidationError("Position must have title and description")
        
        min_exp, max_exp = self.experience_range
        if min_exp < 0:
            raise ValidationError("Minimum experience cannot be negative")
        if max_exp < min_exp:
            raise ValidationError("Maximum experience cannot be less than minimum")
        
        if self.closing_date and self.closing_date < self.posted_date:
            raise ValidationError("Closing date cannot be before posted date")
    
    @classmethod
    def create(
        cls,
        title: str,
        company_id: CompanyId,
        department: str,
        location: Location,
        description: str,
        experience_level: ExperienceLevel,
        work_mode: WorkMode = WorkMode.ONSITE,
        employment_type: EmploymentType = EmploymentType.FULL_TIME,
        responsibilities: Optional[List[str]] = None,
        requirements: Optional[List[str]] = None,
        preferred_qualifications: Optional[List[str]] = None,
        required_skills: Optional[SkillSet] = None,
        preferred_skills: Optional[SkillSet] = None,
        min_experience_years: float = 0,
        max_experience_years: Optional[float] = None,
        education_requirements: Optional[List[EducationLevel]] = None,
        salary_range: Optional[SalaryRange] = None,
        benefits: Optional[List[str]] = None,
        closing_date: Optional[datetime] = None,
        created_by: Optional[UserId] = None,
    ) -> Position:
        """Factory method to create position."""
        position_id = PositionId(uuid4())
        
        # Default max experience based on level if not provided
        if max_experience_years is None:
            max_experience_years = min_experience_years + 5
        
        position = cls(
            id=position_id,
            audit_info=AuditInfo(
                created_at=datetime.now(),
                created_by=created_by,
                updated_at=None,
                updated_by=None,
                version=1,
            ),
            title=title,
            company_id=company_id,
            department=department,
            location=location,
            work_mode=work_mode,
            employment_type=employment_type,
            description=description,
            responsibilities=responsibilities or [],
            requirements=requirements or [],
            preferred_qualifications=preferred_qualifications or [],
            required_skills=required_skills or SkillSet(),
            preferred_skills=preferred_skills or SkillSet(),
            experience_level=experience_level,
            experience_range=(min_experience_years, max_experience_years),
            education_requirements=education_requirements or [],
            salary_range=salary_range,
            benefits=benefits or [],
            is_active=True,
            posted_date=datetime.now(),
            closing_date=closing_date,
        )
        
        # Add creation event
        position.add_event(PositionCreated(
            position_id=position_id,
            company_id=company_id,
            title=title,
            location=location.formatted(),
        ))
        
        return position
    
    def deactivate(self, updated_by: Optional[UserId] = None) -> None:
        """Deactivate position."""
        if not self.is_active:
            raise BusinessRuleViolation("Position is already inactive")
        
        self.is_active = False
        self.mark_updated(updated_by)
        
        self.add_event(PositionUpdated(
            position_id=PositionId(self.id),
            fields_updated=["is_active"],
        ))
    
    def extend_closing_date(self, new_date: datetime, updated_by: Optional[UserId] = None) -> None:
        """Extend position closing date."""
        if not self.is_active:
            raise BusinessRuleViolation("Cannot extend closing date for inactive position")
        
        if self.closing_date and new_date <= self.closing_date:
            raise BusinessRuleViolation("New closing date must be after current closing date")
        
        self.closing_date = new_date
        self.mark_updated(updated_by)
        
        self.add_event(PositionUpdated(
            position_id=PositionId(self.id),
            fields_updated=["closing_date"],
        ))
    
    def is_remote_friendly(self) -> bool:
        """Check if position supports remote work."""
        return self.work_mode in [WorkMode.REMOTE, WorkMode.HYBRID]
    
    def days_open(self) -> int:
        """Calculate how many days position has been open."""
        end_date = self.closing_date if not self.is_active else datetime.now()
        return (end_date - self.posted_date).days


@dataclass
class Match(Entity):
    """Match entity representing a resume-position match."""
    
    resume_id: ResumeId
    position_id: PositionId
    score_details: MatchScoreDetails
    strengths: List[str]
    gaps: List[str]
    recommendations: List[str]
    notes: Optional[str]
    
    def _validate_invariants(self) -> None:
        """Validate match invariants."""
        super()._validate_invariants()
        
        if not self.resume_id or not self.position_id:
            raise ValidationError("Match must have both resume and position IDs")
    
    @classmethod
    def create(
        cls,
        resume: Resume,
        position: Position,
        score_details: MatchScoreDetails,
        created_by: Optional[UserId] = None,
    ) -> Match:
        """Factory method to create match with analysis."""
        # Analyze match
        strengths = cls._analyze_strengths(resume, position, score_details)
        gaps = cls._analyze_gaps(resume, position, score_details)
        recommendations = cls._generate_recommendations(resume, position, score_details, gaps)
        
        match = cls(
            id=uuid4(),
            audit_info=AuditInfo(
                created_at=datetime.now(),
                created_by=created_by,
                updated_at=None,
                updated_by=None,
                version=1,
            ),
            resume_id=ResumeId(resume.id),
            position_id=PositionId(position.id),
            score_details=score_details,
            strengths=strengths,
            gaps=gaps,
            recommendations=recommendations,
            notes=None,
        )
        
        # Add event
        match.add_event(MatchCalculated(
            match_id=match.id,
            resume_id=ResumeId(resume.id),
            position_id=PositionId(position.id),
            overall_score=score_details.overall_score,
            confidence=score_details.confidence,
        ))
        
        return match
    
    @staticmethod
    def _analyze_strengths(resume: Resume, position: Position, scores: MatchScoreDetails) -> List[str]:
        """Analyze and return match strengths."""
        strengths = []
        
        # Check score-based strengths
        strengths.extend(scores.get_strengths())
        
        # Additional analysis
        if resume.inferred_experience_level() == position.experience_level:
            strengths.append("Experience level perfectly matches requirements")
        
        if resume.work_mode_preference == position.work_mode:
            strengths.append("Work mode preference aligns with position")
        
        # Check for preferred skills
        preferred_matched = resume.skills.match_percentage(
            {s.name for s in position.preferred_skills}
        )
        if preferred_matched > 50:
            strengths.append(f"Has {preferred_matched:.0f}% of preferred skills")
        
        return strengths
    
    @staticmethod
    def _analyze_gaps(resume: Resume, position: Position, scores: MatchScoreDetails) -> List[str]:
        """Analyze and return match gaps."""
        gaps = []
        
        # Check score-based weaknesses
        gaps.extend(scores.get_weaknesses())
        
        # Skills analysis
        missing_required = {s.name for s in position.required_skills} - {s.name for s in resume.skills}
        if missing_required:
            gaps.append(f"Missing required skills: {', '.join(list(missing_required)[:3])}")
        
        # Experience analysis
        total_exp = resume.total_experience_years()
        min_exp, _ = position.experience_range
        if total_exp < min_exp:
            gaps.append(f"Needs {min_exp - total_exp:.1f} more years of experience")
        
        # Education analysis
        if position.education_requirements:
            highest_edu = resume.highest_education_level()
            if not highest_edu or highest_edu < min(position.education_requirements):
                gaps.append("Education requirements not met")
        
        return gaps
    
    @staticmethod
    def _generate_recommendations(
        resume: Resume,
        position: Position,
        scores: MatchScoreDetails,
        gaps: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Based on overall score
        if scores.overall_score > 0.8:
            recommendations.append("Strong candidate - recommend immediate interview")
        elif scores.overall_score > 0.6:
            recommendations.append("Good candidate - recommend phone screening")
        else:
            recommendations.append("Consider for future opportunities")
        
        # Skills recommendations
        if scores.skill_score < 0.7:
            recommendations.append("Assess transferable skills and learning ability")
        
        # Experience recommendations
        if scores.experience_score < 0.6:
            recommendations.append("Evaluate potential for growth into the role")
        
        # Salary recommendations
        if scores.salary_score < 0.8:
            recommendations.append("Discuss compensation expectations early in process")
        
        # Interview focus areas
        if gaps:
            recommendations.append(f"Interview focus: {gaps[0]}")
        
        return recommendations
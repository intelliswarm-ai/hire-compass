from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class EducationLevel(str, Enum):
    HIGH_SCHOOL = "high_school"
    BACHELORS = "bachelors"
    MASTERS = "masters"
    PHD = "phd"
    OTHER = "other"

class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"

class WorkMode(str, Enum):
    ONSITE = "onsite"
    REMOTE = "remote"
    HYBRID = "hybrid"

class Education(BaseModel):
    degree: str
    field: str
    institution: str
    graduation_year: Optional[int] = None
    level: EducationLevel

class Experience(BaseModel):
    company: str
    position: str
    duration_months: int
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    description: str
    technologies: List[str] = []

class Skill(BaseModel):
    name: str
    proficiency: Optional[str] = None
    years_of_experience: Optional[float] = None

class Resume(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    education: List[Education] = []
    experience: List[Experience] = []
    skills: List[Skill] = []
    certifications: List[str] = []
    languages: List[str] = []
    total_experience_years: float
    current_salary: Optional[float] = None
    expected_salary: Optional[float] = None
    raw_text: str
    parsed_at: datetime = Field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None

class JobPosition(BaseModel):
    id: str
    title: str
    department: str
    location: str
    work_mode: WorkMode
    description: str
    responsibilities: List[str] = []
    requirements: List[str] = []
    preferred_qualifications: List[str] = []
    required_skills: List[str] = []
    preferred_skills: List[str] = []
    experience_level: ExperienceLevel
    min_experience_years: float
    max_experience_years: Optional[float] = None
    education_requirements: List[EducationLevel] = []
    salary_range_min: Optional[float] = None
    salary_range_max: Optional[float] = None
    posted_date: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    embedding: Optional[List[float]] = None

class AspirationProfile(BaseModel):
    desired_roles: List[str] = []
    desired_industries: List[str] = []
    preferred_company_size: Optional[str] = None
    work_mode_preference: Optional[WorkMode] = None
    location_preferences: List[str] = []
    career_goals: Optional[str] = None
    growth_areas: List[str] = []
    avoid_areas: List[str] = []

class SalaryResearch(BaseModel):
    position_title: str
    location: str
    experience_level: ExperienceLevel
    market_average: float
    market_min: float
    market_max: float
    sources: List[Dict[str, Any]] = []
    researched_at: datetime = Field(default_factory=datetime.now)

class MatchResult(BaseModel):
    resume_id: str
    position_id: str
    overall_score: float
    skill_match_score: float
    experience_match_score: float
    education_match_score: float
    aspiration_match_score: float
    salary_compatibility_score: float
    strengths: List[str] = []
    gaps: List[str] = []
    recommendations: List[str] = []
    detailed_analysis: Dict[str, Any] = {}
    matched_at: datetime = Field(default_factory=datetime.now)

class BatchMatchRequest(BaseModel):
    resume_ids: List[str]
    position_ids: List[str]
    include_salary_research: bool = True
    include_aspiration_analysis: bool = True

class BatchMatchResponse(BaseModel):
    results: List[MatchResult]
    processing_time_seconds: float
    total_comparisons: int
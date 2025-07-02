# Developer Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Code Structure](#code-structure)
4. [Development Workflow](#development-workflow)
5. [Coding Standards](#coding-standards)
6. [Adding New Features](#adding-new-features)
7. [Testing Guidelines](#testing-guidelines)
8. [Debugging](#debugging)
9. [Performance Optimization](#performance-optimization)
10. [Best Practices](#best-practices)

## Getting Started

### Prerequisites

Before starting development, ensure you have:

- Python 3.9 or higher
- Git
- Docker and Docker Compose
- VSCode or PyCharm (recommended IDEs)
- Ollama installed locally
- PostgreSQL (for local development)
- Redis (for caching)

### Initial Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/hire-compass.git
cd hire-compass
```

2. **Set up Python environment:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

3. **Configure environment:**
```bash
# Copy example environment file
cp env.example .env

# Edit .env with your local settings
# Important variables to set:
# - OLLAMA_BASE_URL=http://localhost:11434
# - DB_POSTGRES_PASSWORD=your_local_password
# - ENVIRONMENT=development
# - DEBUG=true
```

4. **Set up pre-commit hooks:**
```bash
pre-commit install
```

5. **Initialize local services:**
```bash
# Start Ollama
ollama serve

# Pull required models
ollama pull llama2

# Start PostgreSQL and Redis (using Docker)
docker-compose up -d postgres redis

# Initialize database
python scripts/init_db.py
```

## Development Environment

### Recommended IDE Setup

#### VSCode

1. **Install extensions:**
   - Python
   - Pylance
   - Python Docstring Generator
   - GitLens
   - Docker
   - Thunder Client (API testing)

2. **VSCode settings.json:**
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=100"],
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

#### PyCharm

1. **Configure interpreter:**
   - Set virtual environment as project interpreter
   - Enable Django support (for better code completion)

2. **Configure code style:**
   - Set line length to 100
   - Enable optimize imports on save
   - Configure docstring format (Google style)

### Development Tools

```bash
# Install development tools
pip install black isort flake8 mypy pytest pytest-asyncio pytest-cov

# Format code
black . --line-length 100

# Sort imports
isort .

# Lint code
flake8 .

# Type checking
mypy .

# Run tests
pytest
```

## Code Structure

### Project Layout

```
hire-compass/
├── api/                    # API layer
│   ├── endpoints/         # API endpoint definitions
│   ├── middleware/        # Custom middleware
│   ├── dependencies/      # FastAPI dependencies
│   └── async_main.py     # Main async API entry
│
├── agents/                # Agent implementations
│   ├── base_agent.py     # Base agent class
│   ├── async_*.py        # Async agent variants
│   └── protocols.py      # Agent protocols
│
├── src/                   # Core business logic (Clean Architecture)
│   ├── domain/           # Domain entities and logic
│   │   ├── entities/    # Domain entities
│   │   ├── value_objects/ # Value objects
│   │   ├── events/      # Domain events
│   │   └── services/    # Domain services
│   │
│   ├── application/      # Application layer
│   │   ├── use_cases/   # Use case implementations
│   │   ├── dto/         # Data transfer objects
│   │   └── interfaces/  # Port interfaces
│   │
│   ├── infrastructure/   # Infrastructure layer
│   │   ├── adapters/    # External service adapters
│   │   ├── persistence/ # Database implementations
│   │   └── config/      # Configuration
│   │
│   └── shared/          # Shared kernel
│       ├── types.py     # Shared types
│       └── protocols.py # Shared protocols
│
├── tools/                # Utilities and tools
│   ├── async_*.py       # Async tool implementations
│   └── helpers/         # Helper functions
│
├── mcp_server/          # MCP server implementations
│   └── servers/         # Individual MCP servers
│
├── tests/               # Test suite
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── e2e/            # End-to-end tests
│
└── scripts/             # Utility scripts
    ├── init_db.py      # Database initialization
    └── migrate.py      # Migration scripts
```

### Architecture Patterns

#### Clean Architecture

```python
# Domain Entity (src/domain/entities/resume.py)
from dataclasses import dataclass
from typing import List
from src.shared.types import ResumeId, Email

@dataclass
class Resume:
    id: ResumeId
    name: str
    email: Email
    skills: List[Skill]
    
    def add_skill(self, skill: Skill) -> None:
        """Business logic for adding skills"""
        if skill not in self.skills:
            self.skills.append(skill)

# Use Case (src/application/use_cases/match_resume.py)
from src.domain.repositories import ResumeRepository, PositionRepository
from src.application.dto import MatchRequestDTO, MatchResponseDTO

class MatchResumeUseCase:
    def __init__(
        self,
        resume_repo: ResumeRepository,
        position_repo: PositionRepository,
        matcher: MatchingService
    ):
        self.resume_repo = resume_repo
        self.position_repo = position_repo
        self.matcher = matcher
    
    async def execute(self, request: MatchRequestDTO) -> MatchResponseDTO:
        resume = await self.resume_repo.get_by_id(request.resume_id)
        position = await self.position_repo.get_by_id(request.position_id)
        
        match_result = await self.matcher.match(resume, position)
        
        return MatchResponseDTO.from_domain(match_result)

# Infrastructure Implementation (src/infrastructure/persistence/postgres_resume_repo.py)
from src.domain.repositories import ResumeRepository
from src.domain.entities import Resume

class PostgresResumeRepository(ResumeRepository):
    def __init__(self, connection_pool):
        self.pool = connection_pool
    
    async def get_by_id(self, resume_id: ResumeId) -> Resume:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM resumes WHERE id = $1", 
                resume_id
            )
            return self._to_domain(row)
```

#### Dependency Injection

```python
# src/infrastructure/config/container.py
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    # Configuration
    config = providers.Configuration()
    
    # Infrastructure
    db_pool = providers.Singleton(
        create_db_pool,
        dsn=config.database.dsn
    )
    
    # Repositories
    resume_repository = providers.Singleton(
        PostgresResumeRepository,
        connection_pool=db_pool
    )
    
    # Services
    matching_service = providers.Factory(
        MatchingService,
        vector_store=providers.Singleton(VectorStore)
    )
    
    # Use Cases
    match_resume_use_case = providers.Factory(
        MatchResumeUseCase,
        resume_repo=resume_repository,
        position_repo=position_repository,
        matcher=matching_service
    )
```

## Development Workflow

### Git Workflow

1. **Feature Development:**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new feature"

# Push to remote
git push origin feature/your-feature-name

# Create pull request
```

2. **Commit Message Convention:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Build/auxiliary tool changes

Example:
```
feat(agents): add async resume parser

- Implemented AsyncResumeParserAgent class
- Added concurrent parsing support
- Improved performance by 5x

Closes #123
```

### Code Review Process

1. **Before submitting PR:**
   - Run all tests: `pytest`
   - Check code style: `black . && isort . && flake8`
   - Update documentation
   - Add/update tests

2. **PR Checklist:**
   - [ ] Tests pass
   - [ ] Code follows style guide
   - [ ] Documentation updated
   - [ ] No security vulnerabilities
   - [ ] Performance impact considered

## Coding Standards

### Python Style Guide

1. **General Rules:**
   - Follow PEP 8 with 100-character line limit
   - Use type hints for all functions
   - Write descriptive variable names
   - Avoid abbreviations

2. **Imports:**
```python
# Standard library imports
import os
import sys
from typing import List, Dict, Optional

# Third-party imports
import numpy as np
from fastapi import FastAPI

# Local imports
from src.domain.entities import Resume
from src.shared.types import ResumeId
```

3. **Type Hints:**
```python
from typing import List, Dict, Optional, Union
from src.shared.types import ResumeId, Score

async def match_resumes(
    resume_ids: List[ResumeId],
    position_id: str,
    threshold: Optional[Score] = None
) -> Dict[ResumeId, Score]:
    """Match multiple resumes against a position.
    
    Args:
        resume_ids: List of resume IDs to match
        position_id: Target position ID
        threshold: Minimum score threshold
        
    Returns:
        Dictionary mapping resume IDs to match scores
    """
    ...
```

4. **Error Handling:**
```python
from src.shared.exceptions import ResumeNotFoundError, MatchingError

async def process_resume(resume_id: str) -> Resume:
    try:
        resume = await self.repository.get_by_id(resume_id)
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise ResumeNotFoundError(f"Resume {resume_id} not found") from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise MatchingError("Failed to process resume") from e
    
    return resume
```

### Async Best Practices

1. **Use async/await properly:**
```python
# Good
async def fetch_data():
    results = await asyncio.gather(
        fetch_resume(id1),
        fetch_resume(id2),
        fetch_resume(id3)
    )
    return results

# Bad - sequential execution
async def fetch_data():
    result1 = await fetch_resume(id1)
    result2 = await fetch_resume(id2)
    result3 = await fetch_resume(id3)
    return [result1, result2, result3]
```

2. **Handle concurrent limits:**
```python
async def process_batch(items: List[str], max_concurrent: int = 10):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_limit(item):
        async with semaphore:
            return await process_item(item)
    
    tasks = [process_with_limit(item) for item in items]
    return await asyncio.gather(*tasks)
```

## Adding New Features

### Adding a New Agent

1. **Create agent class:**
```python
# agents/skill_analyzer_agent.py
from agents.base_agent import BaseAgent
from typing import Dict, Any

class SkillAnalyzerAgent(BaseAgent):
    """Agent for analyzing and categorizing skills."""
    
    def __init__(self):
        super().__init__("SkillAnalyzer", model_name="llama2")
        self.skill_taxonomy = self._load_skill_taxonomy()
    
    def create_prompt(self) -> str:
        return """Analyze the following skills and categorize them..."""
    
    def create_tools(self) -> list:
        return [
            self._create_categorize_tool(),
            self._create_similarity_tool()
        ]
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        skills = input_data.get("skills", [])
        
        # Analyze skills
        categorized = await self._categorize_skills(skills)
        similarities = await self._find_similar_skills(skills)
        
        return {
            "success": True,
            "categories": categorized,
            "similarities": similarities
        }
```

2. **Register with orchestrator:**
```python
# agents/orchestrator_agent.py
def __init__(self):
    super().__init__("Orchestrator")
    self.skill_analyzer = SkillAnalyzerAgent()  # Add new agent
```

3. **Add tests:**
```python
# tests/unit/test_skill_analyzer_agent.py
import pytest
from agents.skill_analyzer_agent import SkillAnalyzerAgent

@pytest.mark.asyncio
async def test_skill_categorization():
    agent = SkillAnalyzerAgent()
    result = await agent.process({
        "skills": ["Python", "TensorFlow", "Docker"]
    })
    
    assert result["success"] == True
    assert "categories" in result
    assert len(result["categories"]) == 3
```

### Adding a New API Endpoint

1. **Create endpoint:**
```python
# api/endpoints/skills.py
from fastapi import APIRouter, Depends, HTTPException
from src.application.use_cases import AnalyzeSkillsUseCase
from api.dependencies import get_container

router = APIRouter(prefix="/skills", tags=["skills"])

@router.post("/analyze")
async def analyze_skills(
    skills: List[str],
    container = Depends(get_container)
):
    """Analyze and categorize skills."""
    use_case = container.analyze_skills_use_case()
    
    try:
        result = await use_case.execute(skills)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

2. **Register router:**
```python
# api/async_main.py
from api.endpoints import skills

app.include_router(skills.router)
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                  # Fast, isolated tests
│   ├── test_entities.py
│   ├── test_use_cases.py
│   └── test_agents.py
├── integration/          # Component integration tests
│   ├── test_api.py
│   ├── test_database.py
│   └── test_vector_store.py
├── e2e/                  # Full system tests
│   └── test_matching_flow.py
├── fixtures/             # Test data
│   ├── resumes/
│   └── positions/
└── conftest.py          # Pytest configuration
```

### Writing Tests

1. **Unit Test Example:**
```python
# tests/unit/test_matching_service.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.domain.services import MatchingService

class TestMatchingService:
    @pytest.fixture
    def matching_service(self):
        vector_store = Mock()
        return MatchingService(vector_store)
    
    @pytest.mark.asyncio
    async def test_calculate_skill_match(self, matching_service):
        resume_skills = ["Python", "Django", "PostgreSQL"]
        position_skills = ["Python", "Django", "MySQL"]
        
        score = await matching_service.calculate_skill_match(
            resume_skills, 
            position_skills
        )
        
        assert score == pytest.approx(0.67, rel=0.01)
```

2. **Integration Test Example:**
```python
# tests/integration/test_resume_flow.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_resume_upload_and_match(
    async_client: AsyncClient,
    test_resume_file,
    test_position_id
):
    # Upload resume
    files = {"file": ("resume.pdf", test_resume_file, "application/pdf")}
    response = await async_client.post("/upload/resume", files=files)
    assert response.status_code == 200
    resume_id = response.json()["id"]
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Perform matching
    response = await async_client.post("/match/single", json={
        "resume_id": resume_id,
        "position_id": test_position_id
    })
    
    assert response.status_code == 200
    assert "match" in response.json()
    assert response.json()["match"]["overall_score"] > 0
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
import asyncio
from httpx import AsyncClient
from api.async_main import app

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def test_resume_data():
    """Sample resume data for testing."""
    return {
        "name": "Test User",
        "email": "test@example.com",
        "skills": [
            {"name": "Python", "level": "Expert"},
            {"name": "FastAPI", "level": "Intermediate"}
        ],
        "experience_years": 5
    }
```

## Debugging

### Debugging Tools

1. **Python Debugger (pdb):**
```python
import pdb

async def complex_function():
    data = await fetch_data()
    pdb.set_trace()  # Breakpoint
    result = process_data(data)
    return result
```

2. **Async Debugging:**
```python
import asyncio

# Enable debug mode
asyncio.run(main(), debug=True)

# Or set environment variable
export PYTHONASYNCIODEBUG=1
```

3. **Logging:**
```python
import logging
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)

async def process_item(item):
    logger.debug(f"Processing item: {item}")
    
    try:
        result = await heavy_computation(item)
        logger.info(f"Successfully processed: {item}")
        return result
    except Exception as e:
        logger.error(f"Error processing {item}", exc_info=True)
        raise
```

### Common Issues and Solutions

1. **Async Context Issues:**
```python
# Problem: Running sync code in async context
def sync_function():
    return asyncio.run(async_function())  # Creates new event loop!

# Solution: Use proper async handling
async def async_wrapper():
    return await async_function()
```

2. **Memory Leaks:**
```python
# Problem: Not closing resources
async def process_files(files):
    for file in files:
        f = open(file)  # File handle leak!
        data = f.read()

# Solution: Use context managers
async def process_files(files):
    for file in files:
        async with aiofiles.open(file) as f:
            data = await f.read()
```

## Performance Optimization

### Profiling

1. **CPU Profiling:**
```python
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run your code
    result = expensive_function()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

2. **Async Profiling:**
```python
import aiomonitor
import asyncio

async def main():
    # Start monitoring
    async with aiomonitor.start_monitor(loop=asyncio.get_event_loop()):
        await your_async_application()
```

3. **Memory Profiling:**
```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    large_list = [i for i in range(1000000)]
    return sum(large_list)
```

### Optimization Techniques

1. **Batch Processing:**
```python
# Instead of individual queries
for id in ids:
    result = await fetch_by_id(id)

# Use batch query
results = await fetch_by_ids(ids)
```

2. **Connection Pooling:**
```python
# Create connection pool
async def create_pool():
    return await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=10,
        max_size=20,
        max_queries=50000,
        max_inactive_connection_lifetime=300
    )
```

3. **Caching:**
```python
from functools import lru_cache
from aiocache import cached

@cached(ttl=300)  # Cache for 5 minutes
async def expensive_calculation(param):
    result = await complex_computation(param)
    return result
```

## Best Practices

### Security

1. **Input Validation:**
```python
from pydantic import BaseModel, validator

class ResumeUpload(BaseModel):
    file_size: int
    file_type: str
    
    @validator('file_size')
    def validate_size(cls, v):
        if v > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError('File too large')
        return v
    
    @validator('file_type')
    def validate_type(cls, v):
        allowed = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
        if v not in allowed:
            raise ValueError('Invalid file type')
        return v
```

2. **SQL Injection Prevention:**
```python
# Bad - SQL injection vulnerable
query = f"SELECT * FROM users WHERE id = {user_id}"

# Good - Parameterized query
query = "SELECT * FROM users WHERE id = $1"
result = await conn.fetchrow(query, user_id)
```

### Error Handling

1. **Graceful Degradation:**
```python
async def get_salary_data(job_title: str) -> Optional[Dict]:
    try:
        # Try primary source
        return await fetch_from_primary_api(job_title)
    except PrimaryAPIError:
        logger.warning("Primary API failed, trying secondary")
        try:
            # Fallback to secondary source
            return await fetch_from_secondary_api(job_title)
        except SecondaryAPIError:
            logger.error("All salary APIs failed")
            # Return cached or default data
            return get_cached_salary_data(job_title)
```

2. **Circuit Breaker Pattern:**
```python
from pybreaker import CircuitBreaker

db_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@db_breaker
async def fetch_from_database(query):
    async with get_connection() as conn:
        return await conn.fetch(query)
```

### Documentation

1. **Module Documentation:**
```python
"""
Resume parsing module.

This module provides functionality for parsing various resume formats
and extracting structured information. It supports PDF, DOCX, and 
plain text formats.

Example:
    >>> parser = ResumeParser()
    >>> result = await parser.parse("resume.pdf")
    >>> print(result.name)
    'John Doe'

Attributes:
    SUPPORTED_FORMATS (List[str]): List of supported file extensions

Todo:
    * Add support for RTF format
    * Implement OCR for scanned PDFs
"""
```

2. **API Documentation:**
```python
@router.post("/match", response_model=MatchResult)
async def match_resume_to_position(
    resume_id: str = Query(..., description="UUID of the resume"),
    position_id: str = Query(..., description="UUID of the position"),
    include_details: bool = Query(False, description="Include detailed scoring breakdown")
) -> MatchResult:
    """
    Match a resume against a specific position.
    
    This endpoint performs semantic matching between a resume and job position,
    returning a compatibility score and optional detailed breakdown.
    
    The matching algorithm considers:
    - Skill alignment (40% weight)
    - Experience match (30% weight)
    - Education fit (20% weight)
    - Location compatibility (10% weight)
    
    Returns:
        MatchResult: Object containing overall score and optional details
        
    Raises:
        HTTPException: 404 if resume or position not found
        HTTPException: 500 for internal processing errors
    """
```

This comprehensive developer guide should help new and existing developers understand the codebase, follow best practices, and contribute effectively to the project.
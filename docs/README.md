# HR Matcher - AI-Powered Resume Matching System

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [API Documentation](#api-documentation)
7. [Agent System](#agent-system)
8. [MCP Servers](#mcp-servers)
9. [Performance](#performance)
10. [Configuration](#configuration)
11. [Development Guide](#development-guide)
12. [Testing](#testing)
13. [Deployment](#deployment)
14. [Troubleshooting](#troubleshooting)

## Overview

HR Matcher is an enterprise-grade, AI-powered resume matching system designed to streamline the recruitment process by intelligently matching candidates with job positions at scale. Built with a multi-agent architecture and leveraging state-of-the-art LLMs through Ollama, it can handle up to 300+ positions concurrently.

### Key Technologies

- **Framework**: LangChain with Ollama LLMs
- **Architecture**: Clean Architecture with Domain-Driven Design (DDD)
- **API**: FastAPI with full async support
- **Vector Store**: ChromaDB for semantic search
- **MCP Servers**: FastMCP for extended capabilities
- **Languages**: Python 3.9+

### System Requirements

- Python 3.9 or higher
- Ollama installed and running
- PostgreSQL 12+ (optional, for production)
- Redis 6+ (optional, for caching)
- 8GB+ RAM recommended
- GPU recommended for optimal LLM performance

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                       │
│         (Web UI, Mobile Apps, API Clients, MCP Clients)        │
└─────────────────┬───────────────────────┬──────────────────────┘
                  │                       │
                  ▼                       ▼
┌─────────────────────────┐   ┌─────────────────────────────────┐
│      FastAPI REST       │   │         MCP Servers             │
│         Gateway         │   │  (Resume Analysis, LinkedIn)    │
└───────────┬─────────────┘   └──────────────┬──────────────────┘
            │                                 │
            ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                           │
│              (Coordinates all agent activities)                 │
└───────┬────────┬────────┬────────┬────────┬────────┬──────────┘
        │        │        │        │        │        │
        ▼        ▼        ▼        ▼        ▼        ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
│  Resume  ││   Job    ││ Matching ││  Salary  ││Aspiration│
│  Parser  ││  Parser  ││  Agent   ││  Agent   ││  Agent   │
└──────────┘└──────────┘└──────────┘└──────────┘└──────────┘
        │        │        │        │        │
        ▼        ▼        ▼        ▼        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                         │
├─────────────┬─────────────┬─────────────┬─────────────────────┤
│  ChromaDB   │ PostgreSQL  │    Redis    │    Ollama LLM       │
│   Vector    │  Database   │    Cache    │     Service         │
│    Store    │             │             │                     │
└─────────────┴─────────────┴─────────────┴─────────────────────┘
```

### Clean Architecture Layers

```
┌─────────────────────────────────────────────────┐
│              Presentation Layer                 │
│          (API Endpoints, MCP Servers)          │
├─────────────────────────────────────────────────┤
│              Application Layer                  │
│        (Use Cases, Service Interfaces)         │
├─────────────────────────────────────────────────┤
│               Domain Layer                      │
│      (Entities, Value Objects, Events)         │
├─────────────────────────────────────────────────┤
│           Infrastructure Layer                  │
│   (Repositories, External APIs, Adapters)      │
└─────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User Request
    │
    ▼
API Gateway ──────► Rate Limiter ──────► Authentication
    │                                            │
    ▼                                            ▼
Request Handler ◄──────────────────────── Authorization
    │
    ▼
Orchestrator ──────► Circuit Breaker ──────► Agent Pool
    │                                            │
    ├──► Resume Parser ──────────────────────────┤
    ├──► Job Parser ─────────────────────────────┤
    ├──► Matching Agent ─────────────────────────┤
    ├──► Salary Agent ───────────────────────────┤
    └──► Aspiration Agent ───────────────────────┘
              │
              ▼
         Vector Store ◄────► Cache Layer
              │
              ▼
         Database Layer
```

## Features

### Core Features

- **Multi-Agent AI System**: Specialized agents for different tasks
- **Semantic Matching**: Advanced vector similarity search
- **Batch Processing**: Handle 300+ positions concurrently
- **Real-time Analysis**: Streaming results for immediate feedback
- **Salary Research**: Web scraping for market insights
- **Aspiration Matching**: Employee career goal alignment
- **LinkedIn Integration**: Direct job fetching from companies
- **Kaggle Dataset Support**: Integration with resume datasets

### Advanced Features

- **Async/Await Throughout**: High-performance concurrent operations
- **Circuit Breakers**: Fault tolerance and resilience
- **Distributed Caching**: Redis-based caching with TTL
- **Event Sourcing**: Complete audit trail
- **CQRS Pattern**: Optimized read/write operations
- **Hot Configuration Reload**: Dynamic settings updates
- **Comprehensive Monitoring**: Prometheus metrics and logging

## Installation

### Prerequisites

1. Install Python 3.9+:
```bash
python --version  # Should be 3.9 or higher
```

2. Install Ollama:
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull required model
ollama pull llama2
```

3. Install PostgreSQL (optional for production):
```bash
# macOS
brew install postgresql

# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
```

4. Install Redis (optional for caching):
```bash
# macOS
brew install redis

# Ubuntu/Debian
sudo apt-get install redis-server
```

### Project Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/hire-compass.git
cd hire-compass
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r mcp_server/requirements_linkedin.txt  # For LinkedIn features
```

4. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
python scripts/init_db.py
```

## Quick Start

### 1. Start the Services

```bash
# Terminal 1: Start Ollama (if not running)
ollama serve

# Terminal 2: Start the main API server
python -m uvicorn api.async_main:app --reload

# Terminal 3: Start MCP servers (optional)
python mcp_server/kaggle_resume_server.py
python mcp_server/linkedin_jobs_server.py
```

### 2. Upload a Resume

```bash
curl -X POST "http://localhost:8000/upload/resume" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/resume.pdf"
```

### 3. Upload Job Positions

```bash
curl -X POST "http://localhost:8000/upload/position" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/job_description.pdf"
```

### 4. Perform Matching

```bash
curl -X POST "http://localhost:8000/match/single" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_id": "resume_abc123",
    "position_id": "pos_xyz789",
    "include_salary_research": true,
    "include_aspiration_analysis": true
  }'
```

### 5. Batch Matching

```bash
curl -X POST "http://localhost:8000/match/batch" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_ids": ["resume_1", "resume_2", "resume_3"],
    "position_ids": ["pos_1", "pos_2", "pos_3"],
    "top_k": 10
  }'
```

## API Documentation

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/upload/resume` | Upload resume |
| POST | `/upload/position` | Upload job position |
| POST | `/match/single` | Single resume-position match |
| POST | `/match/batch` | Batch matching |
| POST | `/research/salary` | Salary research |
| GET | `/docs` | Interactive API documentation |

### Detailed API Documentation

Access the interactive API documentation at `http://localhost:8000/docs` when the server is running.

## Agent System

### Available Agents

1. **Orchestrator Agent**
   - Coordinates all other agents
   - Manages workflow and dependencies
   - Handles error recovery and retries

2. **Resume Parser Agent**
   - Extracts structured data from resumes
   - Supports PDF, DOCX, TXT formats
   - Identifies skills, experience, education

3. **Job Parser Agent**
   - Parses job descriptions
   - Extracts requirements and qualifications
   - Identifies key skills and experience levels

4. **Matching Agent**
   - Performs semantic similarity matching
   - Calculates multi-factor match scores
   - Provides detailed match explanations

5. **Salary Research Agent**
   - Aggregates salary data from multiple sources
   - Provides market insights
   - Calculates competitive ranges

6. **Aspiration Agent**
   - Analyzes career goals
   - Matches employee aspirations with positions
   - Suggests career paths

### Agent Communication

```
┌─────────────┐
│ Orchestrator│
└──────┬──────┘
       │ 
   ┌───┴───┐    Message Format:
   │Message│    {
   │ Queue │      "agent": "target_agent",
   └───┬───┘      "action": "process",
       │          "payload": {...},
   ┌───▼────────┬─────────┬──────────┐     "correlation_id": "uuid"
   │Resume      │Job      │Matching  │    }
   │Parser      │Parser   │Agent     │
   └────────────┴─────────┴──────────┘
```

## MCP Servers

### Kaggle Resume MCP Server

Provides advanced resume analysis capabilities:

- **Endpoints**:
  - `/categorize_resume`: ML-based resume categorization
  - `/extract_skills`: Advanced skill extraction
  - `/find_similar_resumes`: Similarity search
  - `/analyze_resume_quality`: Quality scoring

- **Usage**:
```python
# Start the server
python mcp_server/kaggle_resume_server.py

# Use via MCP client
from mcp import Client
client = Client("http://localhost:8000")
result = await client.categorize_resume(resume_text)
```

### LinkedIn Jobs MCP Server

Provides LinkedIn job integration:

- **Endpoints**:
  - `/search_company_jobs`: Fetch jobs from specific companies
  - `/match_resume_to_jobs`: Match resume against LinkedIn jobs
  - `/get_job_trends`: Market trend analysis
  - `/analyze_company`: Company insights

- **Usage**:
```python
# Start the server
python mcp_server/linkedin_jobs_server.py

# Use via MCP client
jobs = await client.search_company_jobs(
    company="Google",
    location="San Francisco"
)
```

## Performance

### Async Performance Metrics

| Operation | Sync Time | Async Time | Speedup |
|-----------|-----------|------------|---------|
| 100 Resume Uploads | 45.2s | 8.3s | 5.4x |
| Vector Store Batch | 23.1s | 4.2s | 5.5x |
| Web Scraping (16 requests) | 18.4s | 3.1s | 5.9x |
| 1000 Matches | 112.3s | 15.7s | 7.2x |

### Optimization Techniques

1. **Connection Pooling**: Reuse database and HTTP connections
2. **Batch Processing**: Process multiple items concurrently
3. **Caching**: Redis cache with smart TTL management
4. **Vector Indexing**: Optimized HNSW indexes in ChromaDB
5. **Async I/O**: Non-blocking operations throughout

### Scaling Recommendations

- **Horizontal Scaling**: Run multiple API instances behind load balancer
- **Database Scaling**: Use read replicas for search operations
- **Cache Scaling**: Redis cluster for distributed caching
- **Vector Store Scaling**: Partition by date or category

## Configuration

### Configuration Sources (Priority Order)

1. Environment variables
2. `.env` file
3. Configuration files (YAML/JSON)
4. Command line arguments
5. Default values

### Key Configuration Options

```yaml
# config.yaml example
app:
  name: "HR Matcher"
  environment: "production"
  debug: false

database:
  postgres:
    host: "localhost"
    port: 5432
    database: "hr_matcher"
    pool_size: 20

cache:
  provider: "redis"
  redis:
    host: "localhost"
    port: 6379
    ttl: 3600

llm:
  provider: "ollama"
  model: "llama2"
  temperature: 0.7
  max_tokens: 2000

matching:
  skill_weight: 0.40
  experience_weight: 0.30
  education_weight: 0.20
  location_weight: 0.10
```

### Environment Variables

See `env.example` for complete list of environment variables.

## Development Guide

### Project Structure

```
hire-compass/
├── api/                    # API endpoints
│   ├── main.py            # Sync API
│   └── async_main.py      # Async API
├── agents/                # AI agents
│   ├── base_agent.py
│   ├── orchestrator_agent.py
│   └── ...
├── src/                   # Clean architecture implementation
│   ├── domain/           # Domain entities
│   ├── application/      # Use cases
│   └── infrastructure/   # External dependencies
├── tools/                # Utilities and tools
├── mcp_server/          # MCP server implementations
├── tests/               # Test suite
├── docs/                # Documentation
└── examples/            # Usage examples
```

### Adding a New Agent

1. Create agent class inheriting from `BaseAgent`:
```python
from agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("CustomAgent")
    
    async def process(self, input_data):
        # Implementation
        pass
```

2. Register with orchestrator:
```python
self.custom_agent = CustomAgent()
```

3. Add to agent routing logic

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write docstrings for classes and methods
- Keep functions under 50 lines
- Use async/await for I/O operations

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_agents.py

# Run async performance tests
python tests/test_async_performance.py
```

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Multi-component interaction
3. **Performance Tests**: Async vs sync comparisons
4. **Load Tests**: High-volume processing

### Writing Tests

```python
import pytest
from agents.resume_parser_agent import ResumeParserAgent

@pytest.mark.asyncio
async def test_resume_parsing():
    agent = ResumeParserAgent()
    result = await agent.process({
        "file_path": "test_resume.pdf"
    })
    assert result["success"] == True
    assert "resume" in result
```

## Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "api.async_main:app", "--host", "0.0.0.0"]
```

```bash
# Build and run
docker build -t hr-matcher .
docker run -p 8000:8000 hr-matcher
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - postgres
      - redis
      - ollama

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: hr_matcher
      POSTGRES_PASSWORD: secret

  redis:
    image: redis:7

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
```

### Production Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Configure proper database credentials
- [ ] Set up SSL/TLS certificates
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerts
- [ ] Configure backup strategy
- [ ] Set up log aggregation
- [ ] Configure auto-scaling

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```
   Error: Cannot connect to Ollama
   Solution: Ensure Ollama is running with `ollama serve`
   ```

2. **Vector Store Error**
   ```
   Error: ChromaDB collection not found
   Solution: Run `python scripts/init_vector_store.py`
   ```

3. **Memory Issues**
   ```
   Error: Out of memory
   Solution: Reduce batch size or increase RAM
   ```

4. **Slow Performance**
   ```
   Issue: Matching takes too long
   Solution: 
   - Use async API endpoints
   - Enable caching
   - Optimize vector store indexes
   ```

### Debug Mode

Enable debug logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

### Health Checks

Monitor system health:
```bash
curl http://localhost:8000/health
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.
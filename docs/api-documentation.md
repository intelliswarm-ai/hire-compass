# API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URLs](#base-urls)
4. [Rate Limiting](#rate-limiting)
5. [Error Handling](#error-handling)
6. [Core Endpoints](#core-endpoints)
7. [MCP Server APIs](#mcp-server-apis)
8. [WebSocket APIs](#websocket-apis)
9. [SDKs and Examples](#sdks-and-examples)

## Overview

The HR Matcher API provides RESTful endpoints for resume matching, job analysis, and recruitment automation. All responses are in JSON format.

### API Version

Current Version: `v1`

### Content Types

- Request: `application/json`
- Response: `application/json`
- File Upload: `multipart/form-data`

## Authentication

### API Key Authentication

Include your API key in the request header:

```bash
curl -H "X-API-Key: your-api-key-here" \
     https://api.hr-matcher.com/v1/health
```

### JWT Authentication

For user-specific operations:

```bash
# Login
curl -X POST https://api.hr-matcher.com/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "password"}'

# Use token
curl -H "Authorization: Bearer your-jwt-token" \
     https://api.hr-matcher.com/v1/profile
```

## Base URLs

| Environment | Base URL |
|------------|----------|
| Production | `https://api.hr-matcher.com/v1` |
| Staging | `https://staging-api.hr-matcher.com/v1` |
| Development | `http://localhost:8000/v1` |

## Rate Limiting

- **Anonymous**: 10 requests/minute
- **Authenticated**: 100 requests/minute
- **Enterprise**: Custom limits

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1640995200
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Resume not found",
    "details": {
      "resume_id": "abc123"
    },
    "request_id": "req_xyz789"
  }
}
```

### HTTP Status Codes

| Status Code | Description |
|------------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Too Many Requests |
| 500 | Internal Server Error |

### Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Request validation failed |
| `RESOURCE_NOT_FOUND` | Requested resource doesn't exist |
| `DUPLICATE_RESOURCE` | Resource already exists |
| `PROCESSING_ERROR` | Error during processing |
| `RATE_LIMIT_EXCEEDED` | Too many requests |

## Core Endpoints

### Health Check

#### GET /health

Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "ollama_status": "healthy",
  "vector_store_status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Resume Management

#### POST /upload/resume

Upload a resume for processing.

**Request:**
```bash
curl -X POST https://api.hr-matcher.com/v1/upload/resume \
     -H "X-API-Key: your-api-key" \
     -F "file=@resume.pdf" \
     -F "metadata={\"source\":\"website\",\"tags\":[\"engineering\"]}"
```

**Response:**
```json
{
  "id": "resume_a1b2c3d4",
  "filename": "john_doe_resume.pdf",
  "status": "processing",
  "message": "Resume uploaded successfully. Processing in background.",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### GET /resumes/{resume_id}

Get resume details.

**Response:**
```json
{
  "id": "resume_a1b2c3d4",
  "name": "John Doe",
  "email": "john.doe@example.com",
  "phone": "+1-555-123-4567",
  "location": "San Francisco, CA",
  "summary": "Experienced software engineer...",
  "total_experience_years": 7.5,
  "skills": [
    {
      "name": "Python",
      "category": "programming",
      "level": "expert",
      "years": 5
    }
  ],
  "experience": [
    {
      "company": "Tech Corp",
      "position": "Senior Software Engineer",
      "start_date": "2020-01-01",
      "end_date": null,
      "current": true,
      "description": "Leading backend development...",
      "technologies": ["Python", "Django", "PostgreSQL"]
    }
  ],
  "education": [
    {
      "degree": "Bachelor of Science",
      "field": "Computer Science",
      "institution": "UC Berkeley",
      "graduation_year": 2016,
      "gpa": 3.8
    }
  ],
  "certifications": [
    {
      "name": "AWS Certified Solutions Architect",
      "issuer": "Amazon",
      "date": "2022-03-15",
      "expires": "2025-03-15"
    }
  ],
  "languages": [
    {
      "language": "English",
      "proficiency": "native"
    },
    {
      "language": "Spanish",
      "proficiency": "intermediate"
    }
  ],
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:31:00Z"
}
```

#### PUT /resumes/{resume_id}

Update resume information.

**Request:**
```json
{
  "location": "New York, NY",
  "expected_salary": {
    "min": 150000,
    "max": 180000,
    "currency": "USD"
  },
  "availability": "2024-02-01",
  "remote_preference": "hybrid"
}
```

#### DELETE /resumes/{resume_id}

Delete a resume.

**Response:**
```json
{
  "message": "Resume deleted successfully",
  "id": "resume_a1b2c3d4"
}
```

### Position Management

#### POST /upload/position

Upload a job position.

**Request:**
```bash
curl -X POST https://api.hr-matcher.com/v1/upload/position \
     -H "X-API-Key: your-api-key" \
     -F "file=@job_description.pdf" \
     -F "metadata={\"department\":\"engineering\",\"urgency\":\"high\"}"
```

**Response:**
```json
{
  "id": "pos_x1y2z3w4",
  "filename": "senior_engineer_position.pdf",
  "status": "processing",
  "message": "Position uploaded successfully. Processing in background.",
  "created_at": "2024-01-15T10:35:00Z"
}
```

#### GET /positions/{position_id}

Get position details.

**Response:**
```json
{
  "id": "pos_x1y2z3w4",
  "title": "Senior Software Engineer",
  "department": "Engineering",
  "location": "San Francisco, CA",
  "remote_allowed": true,
  "description": "We are looking for a senior software engineer...",
  "experience_level": "senior",
  "min_experience_years": 5,
  "max_experience_years": 10,
  "salary_range": {
    "min": 140000,
    "max": 200000,
    "currency": "USD"
  },
  "required_skills": [
    {
      "name": "Python",
      "level": "expert",
      "required": true
    },
    {
      "name": "AWS",
      "level": "intermediate",
      "required": true
    }
  ],
  "preferred_skills": [
    {
      "name": "Kubernetes",
      "level": "intermediate",
      "required": false
    }
  ],
  "responsibilities": [
    "Design and implement scalable backend systems",
    "Lead technical decisions and architecture",
    "Mentor junior developers"
  ],
  "requirements": [
    "BS/MS in Computer Science or equivalent",
    "5+ years of software development experience",
    "Strong problem-solving skills"
  ],
  "benefits": [
    "Competitive salary and equity",
    "Health, dental, and vision insurance",
    "Flexible work arrangements"
  ],
  "created_at": "2024-01-15T10:35:00Z",
  "expires_at": "2024-03-15T23:59:59Z",
  "status": "active"
}
```

### Matching Operations

#### POST /match/single

Match a single resume with a position.

**Request:**
```json
{
  "resume_id": "resume_a1b2c3d4",
  "position_id": "pos_x1y2z3w4",
  "include_salary_research": true,
  "include_aspiration_analysis": true,
  "detailed_scoring": true
}
```

**Response:**
```json
{
  "match_id": "match_m1n2o3p4",
  "resume_id": "resume_a1b2c3d4",
  "position_id": "pos_x1y2z3w4",
  "overall_score": 0.87,
  "match_percentage": 87,
  "recommendation": "strong_match",
  "scoring_breakdown": {
    "skill_match": {
      "score": 0.90,
      "weight": 0.40,
      "details": {
        "matched_skills": ["Python", "AWS", "PostgreSQL"],
        "missing_skills": ["Kubernetes"],
        "additional_skills": ["Django", "React"]
      }
    },
    "experience_match": {
      "score": 0.85,
      "weight": 0.30,
      "details": {
        "years_experience": 7.5,
        "required_range": [5, 10],
        "relevant_experience": 6.0
      }
    },
    "education_match": {
      "score": 0.80,
      "weight": 0.20,
      "details": {
        "degree_match": true,
        "field_match": true,
        "institution_tier": "top_20"
      }
    },
    "location_match": {
      "score": 1.00,
      "weight": 0.10,
      "details": {
        "same_city": true,
        "remote_compatible": true
      }
    }
  },
  "salary_analysis": {
    "candidate_expectation": {
      "min": 150000,
      "max": 180000
    },
    "position_range": {
      "min": 140000,
      "max": 200000
    },
    "market_data": {
      "average": 165000,
      "percentile_25": 145000,
      "percentile_75": 185000,
      "data_points": 250
    },
    "recommendation": "within_range"
  },
  "aspiration_analysis": {
    "career_trajectory_match": 0.85,
    "growth_potential": "high",
    "skill_gap_analysis": [
      {
        "skill": "Kubernetes",
        "current_level": "none",
        "required_level": "intermediate",
        "learning_time_estimate": "3-6 months"
      }
    ],
    "career_path_alignment": "aligned"
  },
  "ai_insights": {
    "strengths": [
      "Strong technical background matches requirements",
      "Experience with similar tech stack",
      "Located in same city as position"
    ],
    "concerns": [
      "Missing Kubernetes experience",
      "Slightly below median years of experience"
    ],
    "recommendations": [
      "Highlight AWS experience in interview",
      "Discuss willingness to learn Kubernetes",
      "Emphasize leadership experience"
    ]
  },
  "created_at": "2024-01-15T10:40:00Z"
}
```

#### POST /match/batch

Batch match multiple resumes with positions.

**Request:**
```json
{
  "resume_ids": [
    "resume_a1b2c3d4",
    "resume_e5f6g7h8",
    "resume_i9j0k1l2"
  ],
  "position_ids": [
    "pos_x1y2z3w4",
    "pos_m3n4o5p6"
  ],
  "filters": {
    "min_score": 0.7,
    "location_match_required": false,
    "experience_range_flexibility": 2
  },
  "top_k": 10,
  "include_details": false
}
```

**Response:**
```json
{
  "batch_id": "batch_b1c2d3e4",
  "matches": [
    {
      "rank": 1,
      "resume_id": "resume_a1b2c3d4",
      "position_id": "pos_x1y2z3w4",
      "overall_score": 0.87,
      "match_percentage": 87
    },
    {
      "rank": 2,
      "resume_id": "resume_e5f6g7h8",
      "position_id": "pos_x1y2z3w4",
      "overall_score": 0.82,
      "match_percentage": 82
    }
  ],
  "summary": {
    "total_comparisons": 6,
    "matches_found": 4,
    "average_score": 0.78,
    "processing_time_ms": 1250
  },
  "created_at": "2024-01-15T10:45:00Z"
}
```

#### GET /match/recommendations/{resume_id}

Get position recommendations for a resume.

**Request Parameters:**
- `limit` (optional): Number of recommendations (default: 10)
- `location` (optional): Filter by location
- `department` (optional): Filter by department
- `min_score` (optional): Minimum match score (default: 0.7)

**Response:**
```json
{
  "resume_id": "resume_a1b2c3d4",
  "recommendations": [
    {
      "position_id": "pos_x1y2z3w4",
      "title": "Senior Software Engineer",
      "company": "Tech Corp",
      "location": "San Francisco, CA",
      "match_score": 0.87,
      "key_matches": ["Python", "AWS", "Team Leadership"],
      "salary_range": {
        "min": 140000,
        "max": 200000
      },
      "posted_date": "2024-01-10",
      "urgency": "high"
    }
  ],
  "insights": {
    "top_matching_skills": ["Python", "AWS", "PostgreSQL"],
    "skill_gaps": ["Kubernetes", "Go"],
    "recommended_locations": ["San Francisco", "Seattle", "Austin"],
    "salary_expectations": {
      "market_average": 165000,
      "recommendation": "Target $160k-$180k"
    }
  }
}
```

### Salary Research

#### POST /research/salary

Research salary data for a position.

**Request:**
```json
{
  "position_title": "Senior Software Engineer",
  "location": "San Francisco, CA",
  "experience_years": 7,
  "skills": ["Python", "AWS", "Kubernetes"],
  "company_size": "large",
  "industry": "technology"
}
```

**Response:**
```json
{
  "position_title": "Senior Software Engineer",
  "location": "San Francisco, CA",
  "salary_data": {
    "base_salary": {
      "min": 140000,
      "max": 220000,
      "median": 175000,
      "average": 178500
    },
    "total_compensation": {
      "min": 180000,
      "max": 350000,
      "median": 250000,
      "average": 265000
    },
    "components": {
      "base": 175000,
      "bonus": 26250,
      "equity": 50000,
      "benefits": 13750
    }
  },
  "market_insights": {
    "demand": "very_high",
    "supply": "moderate",
    "trend": "increasing",
    "yoy_change": 0.08
  },
  "percentiles": {
    "10": 145000,
    "25": 155000,
    "50": 175000,
    "75": 195000,
    "90": 210000
  },
  "factors": {
    "location_multiplier": 1.25,
    "experience_multiplier": 1.15,
    "skill_premium": {
      "Kubernetes": 0.05,
      "AWS": 0.03
    }
  },
  "sources": [
    {
      "name": "Glassdoor",
      "data_points": 450,
      "last_updated": "2024-01-10"
    },
    {
      "name": "Indeed",
      "data_points": 320,
      "last_updated": "2024-01-12"
    }
  ],
  "confidence_score": 0.92,
  "generated_at": "2024-01-15T10:50:00Z"
}
```

### Analytics

#### GET /analytics/matching-trends

Get matching trends over time.

**Request Parameters:**
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)
- `granularity`: `day`, `week`, or `month`

**Response:**
```json
{
  "period": {
    "start": "2024-01-01",
    "end": "2024-01-15"
  },
  "trends": [
    {
      "date": "2024-01-01",
      "matches_created": 125,
      "average_score": 0.76,
      "positions_filled": 8
    }
  ],
  "summary": {
    "total_matches": 1875,
    "average_score": 0.78,
    "top_skills": ["Python", "JavaScript", "AWS"],
    "conversion_rate": 0.12
  }
}
```

## MCP Server APIs

### Kaggle Resume MCP Server

Base URL: `http://localhost:8000` (MCP protocol)

#### categorize_resume

Categorize a resume using ML models.

**MCP Request:**
```json
{
  "method": "categorize_resume",
  "params": {
    "resume_text": "John Doe, Software Engineer...",
    "include_confidence": true
  }
}
```

**MCP Response:**
```json
{
  "primary_category": "Software Development",
  "secondary_categories": [
    "Data Science",
    "DevOps"
  ],
  "confidence_scores": {
    "Software Development": 0.89,
    "Data Science": 0.65,
    "DevOps": 0.45
  },
  "suggested_positions": [
    "Senior Software Engineer",
    "Full Stack Developer",
    "Backend Engineer"
  ]
}
```

#### extract_skills

Extract and categorize skills from resume.

**MCP Request:**
```json
{
  "method": "extract_skills",
  "params": {
    "resume_text": "...Python, Django, React, Docker...",
    "categorize": true,
    "include_proficiency": true
  }
}
```

**MCP Response:**
```json
{
  "skills": [
    {
      "name": "Python",
      "category": "Programming Languages",
      "proficiency": "expert",
      "years_mentioned": 5,
      "context": ["backend development", "data analysis"]
    },
    {
      "name": "Docker",
      "category": "DevOps Tools",
      "proficiency": "intermediate",
      "years_mentioned": 2,
      "context": ["containerization", "deployment"]
    }
  ],
  "skill_summary": {
    "total_skills": 15,
    "categories": {
      "Programming Languages": 4,
      "Frameworks": 3,
      "DevOps Tools": 2,
      "Databases": 2,
      "Other": 4
    }
  }
}
```

### LinkedIn Jobs MCP Server

Base URL: `http://localhost:8002` (MCP protocol)

#### search_company_jobs

Search for jobs at specific companies.

**MCP Request:**
```json
{
  "method": "search_company_jobs",
  "params": {
    "company": "Google",
    "location": "San Francisco, CA",
    "keywords": ["software engineer", "python"],
    "experience_level": "senior",
    "limit": 20
  }
}
```

**MCP Response:**
```json
{
  "company": "Google",
  "jobs": [
    {
      "job_id": "linkedin_job_123",
      "title": "Senior Software Engineer, Cloud Platform",
      "location": "San Francisco, CA",
      "posted_date": "2024-01-10",
      "description": "Join our Cloud Platform team...",
      "requirements": [
        "5+ years software development",
        "Python or Go experience",
        "Distributed systems knowledge"
      ],
      "seniority_level": "Senior",
      "employment_type": "Full-time",
      "job_function": "Engineering",
      "industries": ["Technology", "Cloud Computing"]
    }
  ],
  "metadata": {
    "total_jobs": 45,
    "returned": 20,
    "last_updated": "2024-01-15T10:00:00Z"
  }
}
```

## WebSocket APIs

### Real-time Matching

Connect to real-time matching updates.

**Connection:**
```javascript
const ws = new WebSocket('wss://api.hr-matcher.com/v1/ws/matching');

ws.on('open', () => {
  // Subscribe to updates
  ws.send(JSON.stringify({
    action: 'subscribe',
    filters: {
      position_ids: ['pos_x1y2z3w4'],
      min_score: 0.8
    }
  }));
});

ws.on('message', (data) => {
  const update = JSON.parse(data);
  console.log('New match:', update);
});
```

**Message Format:**
```json
{
  "type": "match_update",
  "data": {
    "match_id": "match_m1n2o3p4",
    "resume_id": "resume_a1b2c3d4",
    "position_id": "pos_x1y2z3w4",
    "score": 0.87,
    "timestamp": "2024-01-15T10:55:00Z"
  }
}
```

### Processing Status

Track processing status in real-time.

**Subscribe:**
```json
{
  "action": "track",
  "resource_type": "resume",
  "resource_id": "resume_a1b2c3d4"
}
```

**Status Updates:**
```json
{
  "type": "status_update",
  "data": {
    "resource_type": "resume",
    "resource_id": "resume_a1b2c3d4",
    "status": "parsing",
    "progress": 45,
    "message": "Extracting skills and experience..."
  }
}
```

## SDKs and Examples

### Python SDK

```python
from hr_matcher import Client

# Initialize client
client = Client(api_key="your-api-key")

# Upload resume
with open("resume.pdf", "rb") as f:
    resume = client.resumes.upload(f, metadata={"source": "website"})

# Upload position
with open("job_description.pdf", "rb") as f:
    position = client.positions.upload(f)

# Perform matching
match = client.matching.single(
    resume_id=resume.id,
    position_id=position.id,
    include_salary_research=True
)

print(f"Match score: {match.overall_score}")
print(f"Recommendation: {match.recommendation}")

# Batch matching
batch_result = client.matching.batch(
    resume_ids=[resume1.id, resume2.id],
    position_ids=[position1.id, position2.id],
    top_k=5
)

for match in batch_result.matches:
    print(f"{match.resume_id} -> {match.position_id}: {match.score}")
```

### JavaScript/TypeScript SDK

```typescript
import { HRMatcherClient } from '@hr-matcher/client';

// Initialize client
const client = new HRMatcherClient({
  apiKey: 'your-api-key'
});

// Upload resume
const resume = await client.resumes.upload(resumeFile, {
  metadata: { source: 'website' }
});

// Real-time matching
const subscription = client.matching.subscribe({
  positionIds: ['pos_123'],
  minScore: 0.8
});

subscription.on('match', (match) => {
  console.log(`New match: ${match.resumeId} (${match.score})`);
});

// Async/await pattern
async function findBestCandidates(positionId: string) {
  const recommendations = await client.matching.getRecommendations(positionId, {
    limit: 10,
    includeDetails: true
  });
  
  return recommendations.candidates;
}
```

### cURL Examples

```bash
# Upload resume
curl -X POST https://api.hr-matcher.com/v1/upload/resume \
  -H "X-API-Key: your-api-key" \
  -F "file=@resume.pdf"

# Match single
curl -X POST https://api.hr-matcher.com/v1/match/single \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_id": "resume_123",
    "position_id": "pos_456"
  }'

# Get recommendations
curl -X GET "https://api.hr-matcher.com/v1/match/recommendations/resume_123?limit=5" \
  -H "X-API-Key: your-api-key"
```

### Postman Collection

Download our [Postman Collection](https://api.hr-matcher.com/docs/postman-collection.json) for easy API testing.

### OpenAPI Specification

Access our OpenAPI 3.0 specification at:
- JSON: `https://api.hr-matcher.com/v1/openapi.json`
- YAML: `https://api.hr-matcher.com/v1/openapi.yaml`

Interactive documentation available at:
- Swagger UI: `https://api.hr-matcher.com/docs`
- ReDoc: `https://api.hr-matcher.com/redoc`

This comprehensive API documentation provides developers with all the information needed to integrate with the HR Matcher system.
# HR Resume Matcher - AI Multi-Agent System

An advanced HR resume matching system that uses multiple AI agents powered by LangChain and Ollama to intelligently match resumes with job positions at scale (up to 300 positions). The system includes salary research, aspiration analysis, and comprehensive matching algorithms.

## Features

- **Multi-Agent Architecture**: Specialized agents for different tasks
  - Resume Parser Agent: Extracts structured data from resumes
  - Job Parser Agent: Processes job descriptions
  - Matching Agent: Performs intelligent resume-position matching
  - Salary Research Agent: Crawls web for market salary data
  - Aspiration Agent: Analyzes career goals and preferences
  - Orchestrator Agent: Coordinates all agents for optimal performance

- **Vector Database**: Efficient similarity search using ChromaDB
- **Scalable Processing**: Handles up to 300 positions with parallel processing
- **REST API**: FastAPI-based interface for easy integration
- **Ollama Support**: Local LLM inference for privacy and cost efficiency

## Architecture

```
┌─────────────────┐
│   HR Client     │
└────────┬────────┘
         │
┌────────▼────────┐
│   FastAPI       │
│   REST API      │
└────────┬────────┘
         │
┌────────▼────────┐
│  Orchestrator   │
│     Agent       │
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┬────────┐
    │         │        │        │        │
┌───▼──┐ ┌───▼──┐ ┌──▼───┐ ┌──▼───┐ ┌──▼───┐
│Resume│ │ Job  │ │Match │ │Salary│ │Aspir.│
│Parser│ │Parser│ │Agent │ │Agent │ │Agent │
└──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
   │        │        │        │        │
   └────────┴────────┴────────┴────────┘
                     │
            ┌────────▼────────┐
            │   ChromaDB      │
            │ Vector Store    │
            └─────────────────┘
```

## Installation

### Prerequisites

1. Python 3.8+
2. Ollama installed and running
3. Chrome/Chromium (for web scraping)

### Setup

1. Clone the repository:
```bash
cd hr-resume-matcher
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install and start Ollama:
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull required models
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

4. Install spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

5. Copy environment configuration:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Start the API Server

```bash
python api/main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Upload Resume
```bash
POST /upload/resume
Content-Type: multipart/form-data
Body: file (PDF, DOCX, or TXT)
```

#### Upload Job Position
```bash
POST /upload/position
Content-Type: multipart/form-data
Body: file (PDF, DOCX, or TXT)
```

#### Single Match
```bash
POST /match/single
Content-Type: application/json
Body: {
  "resume_id": "resume_xxxx",
  "position_id": "pos_xxxx",
  "include_salary_research": true,
  "include_aspiration_analysis": true
}
```

#### Batch Match
```bash
POST /match/batch
Content-Type: application/json
Body: {
  "resume_ids": ["resume_1", "resume_2"],
  "position_ids": ["pos_1", "pos_2", "pos_3"],
  "include_salary_research": true,
  "include_aspiration_analysis": true
}
```

#### Salary Research
```bash
POST /research/salary
Params:
  - position_title: "Senior Software Engineer"
  - location: "San Francisco"
  - experience_years: 8
```

### Example Usage

Run the example script:
```bash
python example_usage.py
```

This will:
1. Create sample resume and job files
2. Test single matching
3. Show match scores and recommendations

### Python SDK Usage

```python
from agents.orchestrator_agent import OrchestratorAgent

# Initialize orchestrator
orchestrator = OrchestratorAgent()

# Single match
result = orchestrator.process_single_match(
    resume_path="path/to/resume.pdf",
    position_path="path/to/job.pdf",
    include_salary=True,
    include_aspirations=True
)

if result["success"]:
    match = result["match"]
    print(f"Overall Score: {match['overall_score']:.2%}")
    print(f"Strengths: {match['strengths']}")
    print(f"Gaps: {match['gaps']}")
```

## Configuration

Edit `.env` file:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest

# Vector Store
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
VECTOR_STORE_COLLECTION=hr_resume_matcher

# Processing
MAX_CONCURRENT_AGENTS=5
SALARY_RESEARCH_TIMEOUT=30

# Web Driver (for salary research)
WEB_DRIVER_PATH=/usr/local/bin/chromedriver
```

## Matching Algorithm

The system uses a weighted scoring system:

- **Skill Match (40%)**: Required and preferred skills alignment
- **Experience Match (30%)**: Years of experience and level
- **Education Match (20%)**: Degree requirements
- **Salary Compatibility (10%)**: Expectations vs. budget

Additional factors:
- Career aspiration alignment
- Location preferences
- Work mode compatibility
- Industry experience

## Performance Optimization

For handling 300+ positions:

1. **Vector Similarity Pre-filtering**: Initial filtering using embeddings
2. **Parallel Processing**: Concurrent agent execution
3. **Batch Operations**: Process multiple matches simultaneously
4. **Caching**: Results cached for repeated queries
5. **Async Processing**: Non-blocking API operations

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Vector Store Issues
```bash
# Clear vector store
rm -rf ./data/chroma_db
```

### Memory Issues
- Reduce `MAX_CONCURRENT_AGENTS` in `.env`
- Process in smaller batches

## Security Considerations

- Resumes contain PII - ensure proper data handling
- Use environment variables for sensitive configuration
- Implement authentication for production use
- Regular cleanup of uploaded files
- GDPR compliance for EU candidates

## MCP Server Extension

The project now includes an MCP (Model Context Protocol) server that provides intelligent resume-to-job-post categorization:

### Resume2Post Features
- AI-powered categorization using hybrid semantic + feature-based approach
- Batch processing for multiple resumes
- Confidence scoring and filtering
- Model training on historical data
- Detailed match explanations

### Quick Start with MCP
```bash
# Start the MCP server
python mcp_server/server.py

# Configure in Claude Desktop
# Add to config: mcp_server path
```

See `mcp_server/README.md` for detailed MCP documentation.

## Future Enhancements

- [ ] Real web scraping for live salary data
- [ ] Interview scheduling integration
- [ ] Candidate communication automation
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Integration with ATS systems

## License

MIT License

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
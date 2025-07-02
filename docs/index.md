# HR Matcher Documentation

Welcome to the comprehensive documentation for HR Matcher - an enterprise-grade, AI-powered resume matching system.

## Quick Navigation

### 📖 Core Documentation

- **[Main README](README.md)** - Complete system overview, features, and getting started guide
- **[Architecture](architecture.md)** - Detailed system architecture with diagrams and component interactions
- **[Developer Guide](developer-guide.md)** - Complete development setup, coding standards, and best practices
- **[Deployment Guide](deployment-guide.md)** - Production deployment instructions for various environments
- **[API Documentation](api-documentation.md)** - Complete REST API reference with examples

### 🏗️ Architecture & Design

#### System Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                       │
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
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌──────────┬────────────┼────────────┬────────────┐
        │          │            │            │            │
        ▼          ▼            ▼            ▼            ▼
  ┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
  │  Resume  ││   Job    ││ Matching ││  Salary  ││LinkedIn  │
  │  Parser  ││  Parser  ││  Agent   ││  Agent   ││  Agent   │
  └──────────┘└──────────┘└──────────┘└──────────┘└──────────┘
```

#### Key Technologies

- **🤖 AI/ML**: LangChain, Ollama LLMs, ChromaDB Vector Store
- **⚡ Performance**: Full async/await implementation, 5x+ performance improvements
- **🏛️ Architecture**: Clean Architecture, DDD, SOLID principles
- **🌐 APIs**: FastAPI, MCP Servers, WebSocket support
- **📊 Data**: PostgreSQL, Redis caching, Vector similarity search
- **☁️ Cloud**: Kubernetes-ready, Docker containers, multi-cloud support

### 🚀 Getting Started

#### Quick Start (5 minutes)
```bash
# Clone and setup
git clone https://github.com/your-org/hire-compass.git
cd hire-compass
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your settings

# Start services
ollama serve  # Terminal 1
uvicorn api.async_main:app --reload  # Terminal 2

# Test the API
curl http://localhost:8000/health
```

#### Development Setup
```bash
# Full development environment
docker-compose -f docker-compose.dev.yml up -d
python scripts/init_db.py
pytest  # Run tests
```

### 🔧 Development

#### Project Structure
```
hire-compass/
├── api/                 # FastAPI endpoints (sync & async)
├── agents/              # AI agents (orchestrator, parsers, matchers)
├── src/                 # Clean architecture implementation
│   ├── domain/         # Entities, value objects, domain services
│   ├── application/    # Use cases, DTOs, interfaces
│   └── infrastructure/ # External adapters, persistence
├── tools/              # Utilities (vector store, web scraper, etc.)
├── mcp_server/         # Model Context Protocol servers
├── tests/              # Comprehensive test suite
├── docs/               # Documentation (you are here!)
└── examples/           # Usage examples and demos
```

#### Code Quality Standards

- **Type Safety**: Full type hints with mypy validation
- **Testing**: >90% test coverage with pytest
- **Code Style**: Black, isort, flake8 compliant
- **Architecture**: Clean Architecture with dependency injection
- **Performance**: Async-first design with concurrent processing

### 📊 Performance Metrics

| Operation | Sync Time | Async Time | Speedup |
|-----------|-----------|------------|---------|
| 100 Resume Uploads | 45.2s | 8.3s | **5.4x** |
| Vector Store Batch | 23.1s | 4.2s | **5.5x** |
| Web Scraping (16 requests) | 18.4s | 3.1s | **5.9x** |
| 1000 Match Operations | 112.3s | 15.7s | **7.2x** |

### 🛠️ API Reference

#### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/upload/resume` | Upload resume file |
| `POST` | `/upload/position` | Upload job position |
| `POST` | `/match/single` | Single resume-position match |
| `POST` | `/match/batch` | Batch matching operations |
| `POST` | `/research/salary` | Salary market research |

#### Example: Single Match
```python
import requests

response = requests.post('http://localhost:8000/match/single', json={
    "resume_id": "resume_abc123",
    "position_id": "pos_xyz789",
    "include_salary_research": True,
    "include_aspiration_analysis": True
})

match = response.json()
print(f"Match Score: {match['overall_score']:.2%}")
print(f"Recommendation: {match['recommendation']}")
```

### 🎯 Use Cases

#### 1. Enterprise Recruitment
- **Scale**: Handle 300+ positions concurrently
- **Accuracy**: Multi-factor matching with explainable AI
- **Integration**: API-first design for HR systems

#### 2. Talent Analytics
- **Market Research**: Real-time salary data aggregation
- **Skill Analysis**: Advanced NLP for skill extraction
- **Trend Analysis**: Career trajectory insights

#### 3. Career Development
- **Aspiration Matching**: Align employee goals with opportunities
- **Skill Gap Analysis**: Identify learning paths
- **Market Intelligence**: Competitive positioning

### 🌐 Deployment Options

#### Local Development
```bash
# Simple development setup
uvicorn api.async_main:app --reload
```

#### Docker Production
```bash
# Multi-container production environment
docker-compose -f docker-compose.prod.yml up -d
```

#### Kubernetes
```bash
# Scalable cloud deployment
helm install hr-matcher ./helm/hr-matcher
kubectl get pods -n hr-matcher
```

#### Cloud Providers
- **AWS**: EKS, RDS, ElastiCache, S3
- **GCP**: GKE, Cloud SQL, Memorystore
- **Azure**: AKS, PostgreSQL, Redis Cache

### 🔐 Security Features

- **Authentication**: JWT tokens, API keys, OAuth2
- **Authorization**: Role-based access control (RBAC)
- **Data Protection**: Encryption at rest and in transit
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: DDoS protection and fair usage
- **Audit Logging**: Complete operation trail

### 📈 Monitoring & Observability

#### Metrics
- **Application**: Request rates, response times, error rates
- **Business**: Match success rates, user engagement
- **Infrastructure**: CPU, memory, disk usage
- **External**: API dependencies, database performance

#### Logging
- **Structured Logging**: JSON format with correlation IDs
- **Log Aggregation**: Centralized with ELK stack
- **Real-time Monitoring**: Grafana dashboards
- **Alerting**: Prometheus + AlertManager

### 🔧 Troubleshooting

#### Common Issues
1. **Ollama Connection**: Ensure Ollama is running (`ollama serve`)
2. **Vector Store**: Initialize ChromaDB (`python scripts/init_vector_store.py`)
3. **Performance**: Use async endpoints for better throughput
4. **Memory**: Adjust batch sizes for large datasets

#### Debug Commands
```bash
# Check system health
curl http://localhost:8000/health

# Enable debug logging
export DEBUG=true LOG_LEVEL=DEBUG

# Run performance tests
python tests/test_async_performance.py
```

### 📚 Additional Resources

#### Examples
- **[Async Integration](../examples/async_integration_example.py)** - High-performance async operations
- **[LinkedIn Usage](../examples/linkedin_usage_example.py)** - LinkedIn job integration
- **[Performance Testing](../tests/test_async_performance.py)** - Benchmarking tools

#### Configuration
- **[Environment Variables](../env.example)** - Complete configuration reference
- **[Docker Compose](../docker-compose.prod.yml)** - Production container setup
- **[Kubernetes Manifests](../k8s/)** - Cloud deployment configurations

#### MCP Servers
- **[Kaggle Resume Server](../mcp_server/kaggle_resume_server.py)** - ML-powered resume analysis
- **[LinkedIn Jobs Server](../mcp_server/linkedin_jobs_server.py)** - Job data integration
- **[Advanced Analyzer](../mcp_server/advanced_resume_analyzer.py)** - NLP resume processing

### 🤝 Contributing

#### Development Process
1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Develop** with tests and documentation
4. **Test** with `pytest` and performance benchmarks
5. **Submit** pull request with detailed description

#### Code Standards
- Follow [Developer Guide](developer-guide.md) conventions
- Maintain >90% test coverage
- Update documentation for new features
- Use async/await for I/O operations

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

### 📞 Support

- **Documentation**: All docs in this `/docs` directory
- **Issues**: [GitHub Issues](https://github.com/your-org/hire-compass/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/hire-compass/discussions)
- **API Support**: `api-support@hr-matcher.com`

---

## Documentation Index

### 📋 Complete File List

1. **[docs/README.md](README.md)** - Main project documentation
2. **[docs/architecture.md](architecture.md)** - System architecture and design
3. **[docs/developer-guide.md](developer-guide.md)** - Development setup and guidelines
4. **[docs/deployment-guide.md](deployment-guide.md)** - Production deployment instructions
5. **[docs/api-documentation.md](api-documentation.md)** - Complete API reference
6. **[docs/index.md](index.md)** - This documentation index

### 🎯 Quick Links by Topic

#### For Developers
- [Development Setup](developer-guide.md#development-environment)
- [Code Structure](developer-guide.md#code-structure)
- [Testing Guidelines](developer-guide.md#testing-guidelines)
- [Adding New Features](developer-guide.md#adding-new-features)

#### For DevOps
- [Docker Deployment](deployment-guide.md#docker-deployment)
- [Kubernetes Setup](deployment-guide.md#kubernetes-deployment)
- [Monitoring Setup](deployment-guide.md#monitoring-and-logging)
- [Security Configuration](deployment-guide.md#production-configuration)

#### For API Users
- [Authentication](api-documentation.md#authentication)
- [Core Endpoints](api-documentation.md#core-endpoints)
- [Error Handling](api-documentation.md#error-handling)
- [SDKs and Examples](api-documentation.md#sdks-and-examples)

#### For Architects
- [High-Level Architecture](architecture.md#system-overview)
- [Component Design](architecture.md#component-architecture)
- [Data Flow](architecture.md#data-flow)
- [Security Architecture](architecture.md#security-architecture)

---

**Last Updated**: January 15, 2024
**Version**: 2.0.0
**Authors**: HR Matcher Development Team
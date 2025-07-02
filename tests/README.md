# HR Resume Matcher - Test Suite

Comprehensive testing suite for validating the HR Resume Matcher system with 300 job positions and ranking accuracy.

## Test Structure

```
tests/
├── data/                    # Test data generation
│   ├── test_data_generator.py
│   ├── job_positions_300.json
│   └── resumes_with_perfect_matches.json
├── unit/                    # Unit tests
│   └── test_ranking_validation.py
├── integration/             # Integration tests
│   └── test_full_pipeline.py
├── performance/             # Performance tests
│   └── test_large_scale_matching.py
├── reports/                 # Test reports
└── run_all_tests.py        # Master test runner
```

## Test Coverage

### 1. **Ranking Validation Tests** (`unit/test_ranking_validation.py`)
- **Skill-based Ranking**: Validates candidates with more matching skills rank higher
- **Experience-based Ranking**: Tests experience level impact on scores
- **Comprehensive Ranking**: Tests all factors combined
- **Ranking Stability**: Ensures consistent rankings across runs
- **Score Distribution**: Validates meaningful score variance
- **Edge Cases**: Tests empty skills, identical candidates

### 2. **Performance Tests** (`performance/test_large_scale_matching.py`)
- **Large Scale Matching**: Single resume vs 300 positions
- **Batch Processing**: Multiple resumes processed efficiently
- **Concurrent Operations**: System behavior under load
- **Memory Efficiency**: Memory usage with large datasets
- **Vector Search Performance**: Initial filtering speed
- **Detailed Match Analysis**: Deep matching for top candidates

### 3. **Integration Tests** (`integration/test_full_pipeline.py`)
- **End-to-End Pipeline**: Complete matching workflow
- **Agent Coordination**: All agents working together
- **Vector Store Integration**: Database operations
- **MCP Server Integration**: Resume categorization
- **Error Handling**: Graceful failure scenarios

## Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_all_tests.py
```

### Run Individual Test Suites
```bash
# Unit tests only
python tests/unit/test_ranking_validation.py

# Performance tests only
python tests/performance/test_large_scale_matching.py

# Integration tests only
python tests/integration/test_full_pipeline.py
```

### Generate Test Data
```bash
# Generate 300 job positions and 50 resumes
python tests/data/test_data_generator.py
```

## Test Data

### Job Positions (300)
- Various departments: Engineering, Data Science, Product, etc.
- Different experience levels: Entry, Mid, Senior, Lead
- Diverse skill requirements
- Multiple locations and work modes
- Realistic salary ranges

### Resumes (50)
- Distribution: 30% junior, 50% mid-level, 20% senior
- Includes 5 "perfect match" resumes for validation
- Varied skills, experience, and education
- Different salary expectations

## Performance Benchmarks

Expected performance metrics with 300 positions:

| Metric | Target | Actual |
|--------|--------|--------|
| Single Resume vs 300 Positions | < 10s | TBD |
| Vector Search Time | < 1s | TBD |
| Detailed Match (Top 20) | < 5s | TBD |
| Batch Processing (10 resumes) | < 30s | TBD |
| Memory Usage | < 500MB | TBD |
| Ranking Accuracy | > 80% | TBD |

## Ranking Validation

The system validates ranking accuracy by:

1. **Perfect Matches**: Pre-planted resumes that should rank in top 5
2. **Skill Alignment**: More matching skills = higher rank
3. **Experience Fit**: Optimal experience ranks higher
4. **Comprehensive Scoring**: All factors properly weighted

### Scoring Weights
- Skill Match: 40%
- Experience Match: 30%
- Education Match: 20%
- Salary Compatibility: 10%

## Test Reports

Reports are generated in multiple formats:

### JSON Report
```json
{
  "test_date": "2024-01-15T10:30:00",
  "overall_statistics": {
    "total_tests": 25,
    "success_rate": 92.0
  },
  "performance_metrics": {
    "total_time_seconds": 45.2,
    "memory_used_mb": 234.5
  }
}
```

### HTML Report
Visual report with:
- Overall statistics dashboard
- Performance charts
- Detailed test results
- Ranking accuracy metrics

## Continuous Integration

### GitHub Actions Configuration
```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python tests/run_all_tests.py
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Start Ollama service
   ollama serve
   ```

2. **Memory Issues**
   - Reduce batch size in tests
   - Clear vector store between tests

3. **Slow Tests**
   - Use subset of positions for quick tests
   - Skip salary research in performance tests

### Debug Mode
```bash
# Run with verbose logging
python tests/run_all_tests.py --verbose
```

## Adding New Tests

1. Create test file in appropriate directory
2. Inherit from `unittest.TestCase`
3. Add to test suite in `run_all_tests.py`
4. Follow naming convention: `test_*.py`

Example:
```python
class NewFeatureTest(unittest.TestCase):
    def test_new_feature(self):
        # Test implementation
        self.assertTrue(result)
```

## Metrics to Monitor

- **Accuracy**: Ranking correctness
- **Performance**: Processing speed
- **Scalability**: Handling 300+ positions
- **Reliability**: Consistent results
- **Resource Usage**: Memory and CPU
# Resume2Post MCP Server

An MCP (Model Context Protocol) server that extends the HR Resume Matcher system with intelligent resume-to-job-post categorization. This server adapts the Resume2Role concept to categorize resumes against specific job posts rather than generic job categories.

## Overview

The Resume2Post MCP server provides AI-powered categorization of resumes to specific job openings using a hybrid approach:
- **Semantic Similarity**: Using sentence transformers to understand content meaning
- **Feature-based Classification**: Using Random Forest on extracted features
- **Ensemble Learning**: Weighted combination of both approaches

## Features

- **Single Resume Categorization**: Match one resume to multiple job posts
- **Batch Processing**: Process multiple resumes in parallel
- **Confidence Scoring**: High/Medium/Low confidence levels for matches
- **Filtering Options**: Filter by location, experience level, confidence
- **Model Training**: Train on historical match data for improved accuracy
- **Match Explanations**: Detailed explanations for why a resume matches a job
- **Configurable Weights**: Adjust semantic vs feature-based matching weights

## Architecture

```
┌────────────────┐
│  MCP Client    │
│ (Claude, etc.) │
└───────┬────────┘
        │
┌───────▼────────┐
│   FastMCP      │
│    Server      │
└───────┬────────┘
        │
┌───────▼────────┐
│ Resume2Post    │
│     Tool       │
└───────┬────────┘
        │
    ┌───┴───┬──────────┬──────────┐
    │       │          │          │
┌───▼──┐ ┌─▼───┐  ┌───▼───┐  ┌───▼───┐
│Text  │ │ML   │  │Vector │  │Existing│
│Prepr.│ │Model│  │Store  │  │Agents │
└──────┘ └─────┘  └───────┘  └───────┘
```

## Installation

1. Install additional dependencies:
```bash
pip install fastmcp transformers sentence-transformers torch joblib
```

2. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Starting the MCP Server

```bash
python mcp_server/server.py
```

### Available Tools

#### 1. categorize_resume
Categorize a single resume to find matching job posts.

```python
{
    "tool": "categorize_resume",
    "arguments": {
        "resume_path": "/path/to/resume.pdf",
        "top_k": 10,
        "min_confidence": "medium",
        "filter_location": "San Francisco",
        "filter_experience_level": "senior"
    }
}
```

#### 2. batch_categorize_resumes
Process multiple resumes simultaneously.

```python
{
    "tool": "batch_categorize_resumes",
    "arguments": {
        "resume_paths": [
            "/path/to/resume1.pdf",
            "/path/to/resume2.pdf"
        ],
        "top_k": 5,
        "parallel": true
    }
}
```

#### 3. train_categorizer
Train the model on historical matching data.

```python
{
    "tool": "train_categorizer",
    "arguments": {
        "match_history_file": "/path/to/history.json",
        "min_score_threshold": 0.7
    }
}
```

#### 4. update_model_weights
Adjust the balance between semantic and feature-based matching.

```python
{
    "tool": "update_model_weights",
    "arguments": {
        "semantic_weight": 0.7,
        "feature_weight": 0.3
    }
}
```

#### 5. explain_match
Get detailed explanation for a specific resume-job match.

```python
{
    "tool": "explain_match",
    "arguments": {
        "resume_path": "/path/to/resume.pdf",
        "job_id": "pos_001"
    }
}
```

## Integration with Claude Desktop

Add to your Claude Desktop configuration:

```json
{
    "mcpServers": {
        "resume2post": {
            "command": "python",
            "args": ["mcp_server/server.py"],
            "cwd": "/path/to/hire-compass"
        }
    }
}
```

## How It Works

### 1. Text Preprocessing
- Cleans and normalizes resume text
- Extracts skills, experience years, education level
- Identifies key sections (summary, experience, education)
- Performs named entity recognition

### 2. Feature Extraction
- **Numerical Features**: Experience years, education level, skill count
- **Binary Features**: Section presence indicators
- **Entity Features**: Organization and location counts
- **Text Statistics**: Word count, text length

### 3. Semantic Analysis
- Converts resume and job descriptions to embeddings
- Uses sentence transformers (all-MiniLM-L6-v2)
- Calculates cosine similarity between embeddings

### 4. Ensemble Scoring
- Combines semantic similarity and feature-based scores
- Default weights: 60% semantic, 40% feature-based
- Configurable through `update_model_weights`

### 5. Confidence Levels
- **High**: Score > 80% (Very strong match)
- **Medium**: Score 60-80% (Good match)
- **Low**: Score < 60% (Potential match)

## Example Conversations with Claude

```
User: "Please categorize the resume at /documents/john_doe.pdf to our open positions"
Claude: [Uses categorize_resume tool to find best matches]

User: "Show me only high-confidence matches in San Francisco"
Claude: [Uses categorize_resume with filters]

User: "Process all resumes in the /candidates folder"
Claude: [Uses batch_categorize_resumes tool]

User: "Why does this resume match the Senior ML Engineer position?"
Claude: [Uses explain_match tool for detailed analysis]
```

## Training the Model

To improve accuracy, train on your historical data:

1. Prepare training data in JSON format:
```json
[
    {
        "resume_data": {
            "id": "resume_001",
            "name": "John Doe",
            "skills": [{"name": "Python"}, {"name": "ML"}],
            "total_experience_years": 8,
            "raw_text": "Full resume text..."
        },
        "job_data": {
            "id": "job_001",
            "title": "Senior ML Engineer",
            "required_skills": ["Python", "TensorFlow"],
            "description": "Job description..."
        },
        "overall_score": 0.85
    }
]
```

2. Train the model:
```bash
# Through MCP
Use the train_categorizer tool with your data file

# Or directly
python -c "from mcp_server.models.resume_categorizer import Resume2PostCategorizer; 
          c = Resume2PostCategorizer(); 
          c.train(your_training_data)"
```

## Performance Optimization

- **Batch Processing**: Use batch_categorize for multiple resumes
- **Parallel Processing**: Enabled by default for batch operations
- **Vector Store Caching**: Leverages existing ChromaDB for fast retrieval
- **Model Persistence**: Trained models are saved and reused

## Troubleshooting

### Model Not Found
If you see "No saved model found", the system will use semantic similarity only until trained.

### Memory Issues
For large batches, process in smaller chunks or reduce `top_k`.

### Slow Processing
- Ensure sentence transformer model is cached
- Use batch processing instead of individual calls
- Consider using GPU if available

## Future Enhancements

- [ ] Active learning from user feedback
- [ ] Multi-language support
- [ ] Custom embedding models
- [ ] Real-time model updates
- [ ] Integration with more MCP clients
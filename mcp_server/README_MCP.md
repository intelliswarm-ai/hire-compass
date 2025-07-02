# MCP Servers for Resume Analysis

This directory contains Model Context Protocol (MCP) servers that extend the HR Matcher system with advanced resume analysis capabilities.

## Servers

### 1. Kaggle Resume Server (`kaggle_resume_server.py`)

A comprehensive MCP server that integrates with the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/) to provide:

#### Features:
- **Resume Category Prediction**: Classifies resumes into 31 job categories using multiple ML models
- **Skills Extraction**: Extracts technical skills, certifications, and soft skills
- **Experience Analysis**: Extracts years of experience and education qualifications
- **Similarity Search**: Finds similar resumes from the dataset
- **Batch Analysis**: Process multiple resumes efficiently
- **Category Insights**: Statistical analysis of job categories

#### Available Tools:
- `analyze_resume`: Comprehensive resume analysis
- `batch_analyze_resumes`: Analyze multiple resumes with filtering
- `find_similar_resumes`: Find similar resumes using TF-IDF
- `get_category_insights`: Get statistics about job categories
- `train_custom_model`: Train classification models on custom data
- `extract_contact_info`: Extract contact details from resumes

### 2. Advanced Resume Analyzer (`advanced_resume_analyzer.py`)

An advanced NLP-powered MCP server that provides deep resume analysis using state-of-the-art models:

#### Features:
- **Semantic Analysis**: Using Sentence Transformers and BERT
- **Skill Ontology**: Hierarchical skill relationships and recommendations
- **Quality Scoring**: Comprehensive resume quality metrics
- **Job Matching**: Semantic similarity between resume and job descriptions
- **Achievement Extraction**: Identifies and categorizes achievements
- **Keyword Optimization**: ATS optimization recommendations

#### Available Tools:
- `analyze_resume_advanced`: Deep NLP-based resume analysis
- `compare_resume_to_job`: Semantic matching with job descriptions
- `generate_skill_recommendations`: Personalized learning paths
- `extract_achievements`: Extract and categorize achievements
- `optimize_resume_keywords`: ATS keyword optimization

## Installation

1. Install Python dependencies:
```bash
pip install -r mcp_server/requirements.txt
```

2. Download required NLP models:
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet
```

3. Download the Kaggle Resume Dataset:
```bash
# Download from: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/
# Place the CSV file in: data/kaggle_resume/resume_dataset.csv
```

## Usage

### Running the Kaggle Resume Server:
```bash
cd mcp_server
python kaggle_resume_server.py
```

### Running the Advanced Resume Analyzer:
```bash
cd mcp_server
python advanced_resume_analyzer.py
```

### Example Client Usage:

```python
from fastmcp import FastMCP

# Connect to Kaggle Resume Server
client = FastMCP("http://localhost:8000")

# Analyze a resume
result = await client.call_tool(
    "analyze_resume",
    resume_text="John Doe, Python Developer with 5 years experience...",
    include_skills=True,
    include_category=True
)

# Find similar resumes
similar = await client.call_tool(
    "find_similar_resumes",
    resume_text="...",
    dataset_path="data/kaggle_resume/resume_dataset.csv",
    top_k=5
)
```

## Model Training

To train custom models on the Kaggle dataset:

```python
# Train models
result = await client.call_tool(
    "train_custom_model",
    dataset_path="data/kaggle_resume/resume_dataset.csv",
    model_type="random_forest",
    save_path="models/kaggle_resume"
)
```

## Architecture

Both servers follow clean architecture principles:

1. **Separation of Concerns**: Clear separation between data processing, ML models, and API
2. **Extensibility**: Easy to add new analysis tools and models
3. **Performance**: Async processing and model caching
4. **Error Handling**: Comprehensive error handling and logging

## Job Categories

The Kaggle dataset includes 31 job categories:
- Software Development (Python, Java, React, etc.)
- Data Science & Analytics
- DevOps & Cloud Engineering
- Business Analysis
- Project Management
- HR & Operations
- Engineering (Mechanical, Electrical, Civil)
- And more...

## Performance Considerations

1. **Model Loading**: Models are loaded once at startup
2. **Batch Processing**: Use batch endpoints for multiple resumes
3. **Caching**: Results are cached where appropriate
4. **Resource Usage**: Deep learning models require significant memory

## Extending the Servers

To add new analysis capabilities:

1. Add new tool functions with the `@mcp.tool()` decorator
2. Implement analysis logic using existing or new models
3. Update the README with new features

## Troubleshooting

1. **Memory Issues**: Reduce batch sizes or use lighter models
2. **Model Loading**: Ensure all required models are downloaded
3. **Performance**: Use GPU acceleration for deep learning models
4. **Dataset Access**: Ensure the Kaggle dataset is properly downloaded

## License

These MCP servers are part of the HR Matcher system and follow the same license terms.
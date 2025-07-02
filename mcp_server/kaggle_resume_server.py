#!/usr/bin/env python3
"""
MCP Server for Kaggle Resume Dataset Integration.

This server provides tools for working with the Kaggle Resume Dataset,
including category prediction, skills extraction, and resume analysis.

Dataset: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/
"""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("kaggle-resume-analyzer")

# Global variables for models and data
resume_data = None
tfidf_vectorizer = None
classifier_models = {}
category_mapping = {
    0: "Peoplesoft resumes",
    1: "SQL Developer/BI Developer",
    2: "React JS Developer",
    3: "Workday Specialist",
    4: "Python Developer",
    5: "DevOps Engineer",
    6: "Hadoop Developer",
    7: "ETL Developer",
    8: "Automation Testing",
    9: "Java Developer",
    10: "Dotnet Developer",
    11: "Sales Executive",
    12: "Testing",
    13: "HR",
    14: "Operations Manager",
    15: "Data Science",
    16: "Marketing",
    17: "Mechanical Engineer",
    18: "Arts",
    19: "Database Administrator",
    20: "Electrical Engineer",
    21: "Health and Fitness",
    22: "PMO",
    23: "Business Analyst",
    24: "DBA Developer",
    25: "Network Engineer",
    26: "Quality Assurance",
    27: "Security Engineer",
    28: "SAP Developer",
    29: "Civil Engineer",
    30: "Web Developer"
}

# Reverse mapping for category names to IDs
category_name_to_id = {v: k for k, v in category_mapping.items()}


class ResumeProcessor:
    """Process and clean resume text."""
    
    @staticmethod
    def clean_resume(resume_text: str) -> str:
        """Clean resume text by removing URLs, special characters, etc."""
        # Remove URLs
        resume_text = re.sub(r'http\S+\s*', ' ', resume_text)
        
        # Remove emails
        resume_text = re.sub(r'\S+@\S+', ' ', resume_text)
        
        # Remove special characters and digits
        resume_text = re.sub(r'[^a-zA-Z\s]', ' ', resume_text)
        
        # Remove extra whitespace
        resume_text = re.sub(r'\s+', ' ', resume_text)
        
        # Convert to lowercase
        resume_text = resume_text.lower().strip()
        
        return resume_text
    
    @staticmethod
    def extract_skills(resume_text: str) -> List[str]:
        """Extract technical skills from resume."""
        # Common technical skills patterns
        skill_patterns = [
            # Programming languages
            r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|rust|kotlin|swift|php|scala|r)\b',
            # Web frameworks
            r'\b(react|angular|vue|django|flask|spring|express|fastapi|rails|laravel)\b',
            # Databases
            r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|cassandra|oracle|dynamodb)\b',
            # Cloud platforms
            r'\b(aws|azure|gcp|google cloud|heroku|digitalocean)\b',
            # DevOps tools
            r'\b(docker|kubernetes|jenkins|gitlab|github|terraform|ansible|chef|puppet)\b',
            # Data science
            r'\b(pandas|numpy|scikit-learn|tensorflow|pytorch|keras|jupyter|tableau|powerbi)\b',
            # Other tools
            r'\b(git|linux|windows|macos|agile|scrum|jira|confluence)\b'
        ]
        
        skills = set()
        resume_lower = resume_text.lower()
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, resume_lower)
            skills.update(matches)
        
        return sorted(list(skills))
    
    @staticmethod
    def extract_experience_years(resume_text: str) -> Optional[float]:
        """Extract years of experience from resume."""
        # Pattern for "X years of experience" or "X+ years"
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*:?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in\s*'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, resume_text.lower())
            if match:
                return float(match.group(1))
        
        return None
    
    @staticmethod
    def extract_education(resume_text: str) -> List[str]:
        """Extract education qualifications."""
        education_patterns = [
            r'\b(bachelor|b\.s\.|b\.a\.|b\.tech|b\.e\.)\b',
            r'\b(master|m\.s\.|m\.a\.|m\.tech|mba|m\.e\.)\b',
            r'\b(phd|ph\.d\.|doctorate)\b',
            r'\b(diploma|certification|certified)\b'
        ]
        
        qualifications = set()
        resume_lower = resume_text.lower()
        
        for pattern in education_patterns:
            if re.search(pattern, resume_lower):
                # Extract the degree type
                if 'bachelor' in pattern or 'b.' in pattern:
                    qualifications.add('Bachelor')
                elif 'master' in pattern or 'm.' in pattern:
                    qualifications.add('Master')
                elif 'phd' in pattern or 'doctorate' in pattern:
                    qualifications.add('PhD')
                elif 'diploma' in pattern:
                    qualifications.add('Diploma')
        
        return sorted(list(qualifications))


def load_models(model_dir: str = "models/kaggle_resume") -> bool:
    """Load pre-trained models."""
    global tfidf_vectorizer, classifier_models
    
    model_path = Path(model_dir)
    if not model_path.exists():
        logger.warning(f"Model directory {model_dir} not found")
        return False
    
    try:
        # Load TF-IDF vectorizer
        tfidf_path = model_path / "tfidf_vectorizer.pkl"
        if tfidf_path.exists():
            tfidf_vectorizer = joblib.load(tfidf_path)
            logger.info("Loaded TF-IDF vectorizer")
        
        # Load classifiers
        for classifier_name in ["knn", "random_forest", "svm"]:
            classifier_path = model_path / f"{classifier_name}_classifier.pkl"
            if classifier_path.exists():
                classifier_models[classifier_name] = joblib.load(classifier_path)
                logger.info(f"Loaded {classifier_name} classifier")
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False


def train_models(data_path: str, save_dir: str = "models/kaggle_resume") -> Dict[str, float]:
    """Train classification models on resume data."""
    global tfidf_vectorizer, classifier_models
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} resumes from {data_path}")
    
    # Clean resumes
    df['cleaned_resume'] = df['Resume'].apply(ResumeProcessor.clean_resume)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['cleaned_resume'] = df['cleaned_resume'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )
    
    # Prepare features and labels
    X = df['cleaned_resume'].values
    y = df['Category'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1500,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train multiple classifiers
    classifiers = {
        'knn': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        logger.info(f"Training {name} classifier...")
        
        # Use OneVsRest for multi-class classification
        ovr_clf = OneVsRestClassifier(clf)
        ovr_clf.fit(X_train_tfidf, y_train)
        
        # Predict and evaluate
        y_pred = ovr_clf.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"{name} accuracy: {accuracy:.4f}")
        results[name] = accuracy
        
        # Store trained model
        classifier_models[name] = ovr_clf
    
    # Save models
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(tfidf_vectorizer, save_path / "tfidf_vectorizer.pkl")
    for name, model in classifier_models.items():
        joblib.dump(model, save_path / f"{name}_classifier.pkl")
    
    logger.info(f"Models saved to {save_dir}")
    return results


@mcp.tool()
async def analyze_resume(
    resume_text: str,
    include_skills: bool = True,
    include_category: bool = True,
    include_experience: bool = True,
    include_education: bool = True
) -> Dict[str, Any]:
    """
    Analyze a resume and extract key information.
    
    Args:
        resume_text: The resume text to analyze
        include_skills: Extract technical skills
        include_category: Predict job category
        include_experience: Extract years of experience
        include_education: Extract education qualifications
    
    Returns:
        Dictionary containing analysis results
    """
    processor = ResumeProcessor()
    results = {"original_length": len(resume_text)}
    
    # Clean resume
    cleaned_text = processor.clean_resume(resume_text)
    results["cleaned_length"] = len(cleaned_text)
    
    # Extract skills
    if include_skills:
        skills = processor.extract_skills(resume_text)
        results["skills"] = skills
        results["skill_count"] = len(skills)
    
    # Extract experience
    if include_experience:
        experience_years = processor.extract_experience_years(resume_text)
        results["experience_years"] = experience_years
    
    # Extract education
    if include_education:
        education = processor.extract_education(resume_text)
        results["education"] = education
    
    # Predict category
    if include_category and tfidf_vectorizer and classifier_models:
        # Remove stopwords for classification
        stop_words = set(stopwords.words('english'))
        cleaned_for_classification = ' '.join([
            word for word in cleaned_text.split() if word not in stop_words
        ])
        
        # Vectorize
        resume_vector = tfidf_vectorizer.transform([cleaned_for_classification])
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in classifier_models.items():
            pred = model.predict(resume_vector)[0]
            pred_proba = model.predict_proba(resume_vector)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(pred_proba)[-3:][::-1]
            top_categories = [
                {
                    "category": category_mapping.get(idx, "Unknown"),
                    "confidence": float(pred_proba[idx])
                }
                for idx in top_indices
            ]
            
            predictions[model_name] = {
                "predicted_category": category_mapping.get(pred, "Unknown"),
                "predicted_id": int(pred),
                "confidence": float(pred_proba[pred]),
                "top_3": top_categories
            }
        
        results["category_predictions"] = predictions
        
        # Ensemble prediction (majority vote)
        all_predictions = [p["predicted_id"] for p in predictions.values()]
        ensemble_pred = max(set(all_predictions), key=all_predictions.count)
        results["ensemble_prediction"] = {
            "category": category_mapping.get(ensemble_pred, "Unknown"),
            "category_id": ensemble_pred
        }
    
    return results


@mcp.tool()
async def batch_analyze_resumes(
    resumes: List[Dict[str, str]],
    target_category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze multiple resumes and optionally filter by category.
    
    Args:
        resumes: List of resume dictionaries with 'id' and 'text' fields
        target_category: Optional category name to filter matches
    
    Returns:
        Analysis results for all resumes
    """
    results = []
    category_distribution = {}
    
    for resume in resumes:
        resume_id = resume.get("id", "unknown")
        resume_text = resume.get("text", "")
        
        # Analyze resume
        analysis = await analyze_resume(
            resume_text,
            include_skills=True,
            include_category=True,
            include_experience=True,
            include_education=True
        )
        
        analysis["resume_id"] = resume_id
        
        # Track category distribution
        if "ensemble_prediction" in analysis:
            category = analysis["ensemble_prediction"]["category"]
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        # Filter by target category if specified
        if target_category:
            if "ensemble_prediction" in analysis:
                if analysis["ensemble_prediction"]["category"] == target_category:
                    results.append(analysis)
        else:
            results.append(analysis)
    
    return {
        "total_analyzed": len(resumes),
        "matches": len(results),
        "category_distribution": category_distribution,
        "results": results
    }


@mcp.tool()
async def find_similar_resumes(
    resume_text: str,
    dataset_path: str,
    top_k: int = 5,
    min_similarity: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Find similar resumes from the dataset.
    
    Args:
        resume_text: The resume to compare
        dataset_path: Path to the resume dataset CSV
        top_k: Number of similar resumes to return
        min_similarity: Minimum similarity score (0-1)
    
    Returns:
        List of similar resumes with similarity scores
    """
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Clean all resumes
    processor = ResumeProcessor()
    cleaned_query = processor.clean_resume(resume_text)
    df['cleaned_resume'] = df['Resume'].apply(processor.clean_resume)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_query = ' '.join([word for word in cleaned_query.split() if word not in stop_words])
    df['cleaned_resume'] = df['cleaned_resume'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    all_resumes = [cleaned_query] + df['cleaned_resume'].tolist()
    tfidf_matrix = vectorizer.fit_transform(all_resumes)
    
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
    
    # Get top similar resumes
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    similar_resumes = []
    for idx in top_indices:
        if similarities[idx] >= min_similarity:
            resume_data = df.iloc[idx]
            similar_resumes.append({
                "index": int(idx),
                "similarity_score": float(similarities[idx]),
                "category": category_mapping.get(resume_data['Category'], "Unknown"),
                "resume_preview": resume_data['Resume'][:500] + "...",
                "skills": processor.extract_skills(resume_data['Resume'])
            })
    
    return similar_resumes


@mcp.tool()
async def get_category_insights(
    dataset_path: str,
    category_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get insights about job categories in the dataset.
    
    Args:
        dataset_path: Path to the resume dataset CSV
        category_name: Optional specific category to analyze
    
    Returns:
        Category statistics and insights
    """
    # Load dataset
    df = pd.read_csv(dataset_path)
    processor = ResumeProcessor()
    
    if category_name:
        # Get specific category ID
        category_id = category_name_to_id.get(category_name)
        if category_id is None:
            return {"error": f"Category '{category_name}' not found"}
        
        # Filter by category
        category_df = df[df['Category'] == category_id]
        
        if category_df.empty:
            return {"error": f"No resumes found for category '{category_name}'"}
        
        # Extract skills for this category
        all_skills = []
        experience_years = []
        education_levels = []
        
        for _, resume in category_df.iterrows():
            skills = processor.extract_skills(resume['Resume'])
            all_skills.extend(skills)
            
            exp = processor.extract_experience_years(resume['Resume'])
            if exp:
                experience_years.append(exp)
            
            edu = processor.extract_education(resume['Resume'])
            education_levels.extend(edu)
        
        # Count skill frequencies
        from collections import Counter
        skill_counts = Counter(all_skills)
        education_counts = Counter(education_levels)
        
        return {
            "category": category_name,
            "resume_count": len(category_df),
            "top_skills": dict(skill_counts.most_common(15)),
            "average_experience": np.mean(experience_years) if experience_years else None,
            "experience_range": {
                "min": min(experience_years) if experience_years else None,
                "max": max(experience_years) if experience_years else None
            },
            "education_distribution": dict(education_counts),
            "sample_size": {
                "with_experience": len(experience_years),
                "with_education": len([e for e in education_levels if e])
            }
        }
    else:
        # Overall dataset insights
        category_counts = df['Category'].value_counts()
        
        insights = {
            "total_resumes": len(df),
            "total_categories": len(category_counts),
            "category_distribution": {}
        }
        
        for cat_id, count in category_counts.items():
            cat_name = category_mapping.get(cat_id, f"Unknown_{cat_id}")
            insights["category_distribution"][cat_name] = {
                "count": int(count),
                "percentage": round((count / len(df)) * 100, 2)
            }
        
        return insights


@mcp.tool()
async def train_custom_model(
    dataset_path: str,
    model_type: str = "random_forest",
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a custom classification model on the resume dataset.
    
    Args:
        dataset_path: Path to the resume dataset CSV
        model_type: Type of model to train (knn, random_forest, svm)
        save_path: Optional path to save the trained model
    
    Returns:
        Training results and metrics
    """
    if model_type not in ["knn", "random_forest", "svm"]:
        return {"error": f"Unsupported model type: {model_type}"}
    
    try:
        results = train_models(dataset_path, save_path or "models/kaggle_resume")
        
        return {
            "status": "success",
            "model_type": model_type,
            "accuracies": results,
            "best_model": max(results, key=results.get),
            "best_accuracy": max(results.values())
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool()
async def extract_contact_info(resume_text: str) -> Dict[str, Any]:
    """
    Extract contact information from resume.
    
    Args:
        resume_text: The resume text
    
    Returns:
        Extracted contact information
    """
    contact_info = {}
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, resume_text)
    if emails:
        contact_info["emails"] = emails
    
    # Extract phone numbers
    phone_patterns = [
        r'\+?1?\s*\(?\d{3}\)?\s*[-.]?\s*\d{3}\s*[-.]?\s*\d{4}',
        r'\d{3}[-.]\d{3}[-.]\d{4}',
        r'\(\d{3}\)\s*\d{3}-\d{4}'
    ]
    
    phones = []
    for pattern in phone_patterns:
        matches = re.findall(pattern, resume_text)
        phones.extend(matches)
    
    if phones:
        contact_info["phones"] = list(set(phones))
    
    # Extract LinkedIn
    linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+'
    linkedin_matches = re.findall(linkedin_pattern, resume_text, re.IGNORECASE)
    if linkedin_matches:
        contact_info["linkedin"] = linkedin_matches[0]
    
    # Extract GitHub
    github_pattern = r'(?:https?://)?(?:www\.)?github\.com/[\w-]+'
    github_matches = re.findall(github_pattern, resume_text, re.IGNORECASE)
    if github_matches:
        contact_info["github"] = github_matches[0]
    
    # Extract name (heuristic - usually at the beginning)
    lines = resume_text.strip().split('\n')
    if lines:
        # First non-empty line often contains the name
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line.split()) <= 4:  # Likely a name
                # Check if it's not an email or phone
                if '@' not in line and not re.search(r'\d{3}', line):
                    contact_info["possible_name"] = line
                    break
    
    return contact_info


# Initialize server on startup
@mcp.on_startup()
async def startup():
    """Initialize models on server startup."""
    logger.info("Kaggle Resume MCP Server starting...")
    
    # Try to load pre-trained models
    models_loaded = load_models()
    
    if not models_loaded:
        logger.warning("Pre-trained models not found. Train models using train_custom_model tool.")
    else:
        logger.info("Pre-trained models loaded successfully.")
    
    logger.info("Server ready to accept requests.")


# Run the server
if __name__ == "__main__":
    mcp.run()
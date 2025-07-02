#!/usr/bin/env python3
"""
Advanced Resume Analyzer MCP Server.

This server provides advanced NLP-based resume analysis using the Kaggle dataset,
including deep learning models, semantic similarity, and comprehensive skill mapping.
"""

import asyncio
import json
import logging
import re
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from fastmcp import FastMCP
import spacy
from spacy.matcher import Matcher, PhraseMatcher
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("advanced-resume-analyzer")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Global variables for models
sentence_model = None
bert_model = None
bert_tokenizer = None


class SkillOntology:
    """Skill ontology for comprehensive skill mapping."""
    
    def __init__(self):
        self.skill_graph = nx.DiGraph()
        self._build_ontology()
    
    def _build_ontology(self):
        """Build skill ontology graph."""
        # Programming languages hierarchy
        programming_skills = {
            "Programming": [
                "Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "Kotlin"
            ],
            "Python": [
                "Django", "Flask", "FastAPI", "Pandas", "NumPy", "Scikit-learn", "PyTorch", "TensorFlow"
            ],
            "JavaScript": [
                "React", "Angular", "Vue.js", "Node.js", "Express.js", "TypeScript"
            ],
            "Java": [
                "Spring", "Spring Boot", "Hibernate", "Maven", "Gradle"
            ]
        }
        
        # Data science hierarchy
        data_skills = {
            "Data Science": [
                "Machine Learning", "Deep Learning", "Statistics", "Data Analysis", "Data Visualization"
            ],
            "Machine Learning": [
                "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "NLP", "Computer Vision"
            ],
            "Deep Learning": [
                "Neural Networks", "CNN", "RNN", "LSTM", "Transformer", "GANs"
            ]
        }
        
        # Cloud and DevOps
        cloud_skills = {
            "Cloud Computing": [
                "AWS", "Azure", "Google Cloud", "Cloud Architecture", "Serverless"
            ],
            "DevOps": [
                "CI/CD", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins"
            ],
            "AWS": [
                "EC2", "S3", "Lambda", "RDS", "DynamoDB", "CloudFormation"
            ]
        }
        
        # Add all hierarchies to graph
        for category, skills in {**programming_skills, **data_skills, **cloud_skills}.items():
            for skill in skills:
                self.skill_graph.add_edge(category, skill)
    
    def get_related_skills(self, skill: str, depth: int = 2) -> Set[str]:
        """Get related skills up to specified depth."""
        related = set()
        
        # Find skill in graph (case-insensitive)
        skill_lower = skill.lower()
        matching_nodes = [n for n in self.skill_graph.nodes() if n.lower() == skill_lower]
        
        if not matching_nodes:
            return {skill}
        
        skill_node = matching_nodes[0]
        
        # Get ancestors (broader skills)
        ancestors = nx.ancestors(self.skill_graph, skill_node)
        related.update(ancestors)
        
        # Get descendants (more specific skills)
        descendants = nx.descendants(self.skill_graph, skill_node)
        related.update(descendants)
        
        # Get siblings (skills at same level)
        for parent in self.skill_graph.predecessors(skill_node):
            siblings = list(self.skill_graph.successors(parent))
            related.update(siblings)
        
        return related
    
    def calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate semantic similarity between two skills."""
        # Check if skills are in the same hierarchy
        related1 = self.get_related_skills(skill1)
        related2 = self.get_related_skills(skill2)
        
        # Calculate Jaccard similarity
        intersection = related1.intersection(related2)
        union = related1.union(related2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)


class ResumeAnalyzer:
    """Advanced resume analysis with NLP."""
    
    def __init__(self):
        self.skill_ontology = SkillOntology()
        self.skill_patterns = self._build_skill_patterns()
    
    def _build_skill_patterns(self) -> Dict[str, List[str]]:
        """Build comprehensive skill patterns."""
        return {
            "technical_skills": [
                # Programming Languages
                r'\b(python|java|javascript|typescript|c\+\+|c#|c sharp|golang|go|rust|kotlin|swift|'
                r'scala|ruby|php|perl|r|matlab|julia|fortran|cobol|pascal|haskell|erlang|elixir)\b',
                
                # Web Technologies
                r'\b(html5?|css3?|sass|scss|less|bootstrap|tailwind|material.ui|ant.design)\b',
                r'\b(react\.?js|angular\.?js|vue\.?js|svelte|next\.?js|nuxt\.?js|gatsby)\b',
                r'\b(node\.?js|express\.?js|koa|fastify|nest\.?js|deno)\b',
                
                # Databases
                r'\b(sql|mysql|postgresql|postgres|mongodb|redis|cassandra|dynamodb|cosmos.?db|'
                r'elastic.?search|neo4j|influxdb|timescaledb|cockroachdb)\b',
                
                # Cloud & DevOps
                r'\b(aws|amazon.web.services|azure|google.cloud|gcp|alibaba.cloud|digitalocean|heroku)\b',
                r'\b(docker|kubernetes|k8s|openshift|rancher|nomad|mesos)\b',
                r'\b(jenkins|gitlab.ci|github.actions|circle.?ci|travis.?ci|bamboo|teamcity)\b',
                r'\b(terraform|ansible|puppet|chef|saltstack|pulumi)\b',
                
                # Data Science & ML
                r'\b(machine.learning|deep.learning|neural.networks?|nlp|natural.language.processing|'
                r'computer.vision|reinforcement.learning)\b',
                r'\b(tensorflow|pytorch|keras|scikit.?learn|xgboost|lightgbm|catboost)\b',
                r'\b(pandas|numpy|scipy|matplotlib|seaborn|plotly|bokeh)\b',
                r'\b(jupyter|anaconda|spyder|rstudio)\b',
                
                # Big Data
                r'\b(hadoop|spark|apache.spark|pyspark|hive|presto|impala|flink|storm|kafka)\b',
                r'\b(airflow|luigi|dagster|prefect|kedro)\b',
                
                # Mobile Development
                r'\b(android|ios|react.native|flutter|xamarin|ionic|cordova)\b',
                r'\b(swift|objective.?c|kotlin|java.android)\b',
                
                # Other Technologies
                r'\b(git|github|gitlab|bitbucket|svn|mercurial)\b',
                r'\b(rest.?api|graphql|grpc|websockets?|microservices|serverless)\b',
                r'\b(agile|scrum|kanban|jira|confluence|asana|trello)\b',
                r'\b(linux|unix|windows.server|macos|ubuntu|centos|debian)\b'
            ],
            
            "soft_skills": [
                r'\b(leadership|team.lead|mentor|coaching|management)\b',
                r'\b(communication|presentation|public.speaking|writing)\b',
                r'\b(problem.solving|analytical|critical.thinking|creative)\b',
                r'\b(collaboration|teamwork|interpersonal|cross.functional)\b',
                r'\b(adaptability|flexibility|quick.learner|self.motivated)\b',
                r'\b(time.management|organized|detail.oriented|multitasking)\b'
            ],
            
            "certifications": [
                r'\b(aws.certified|azure.certified|gcp.certified|cisco.ccna|ccnp|ccie)\b',
                r'\b(pmp|prince2|agile.certified|scrum.master|safe)\b',
                r'\b(cissp|ceh|comptia|security\+|network\+)\b',
                r'\b(oracle.certified|microsoft.certified|vmware.certified)\b',
                r'\b(cfa|cpa|six.sigma|itil)\b'
            ]
        }
    
    def extract_comprehensive_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills using multiple methods."""
        text_lower = text.lower()
        extracted_skills = defaultdict(set)
        
        # Pattern-based extraction
        for skill_type, patterns in self.skill_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                extracted_skills[skill_type].update(matches)
        
        # NLP-based extraction if spaCy is available
        if nlp:
            doc = nlp(text)
            
            # Extract technical terms using POS tagging
            technical_terms = []
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2:
                    if any(char.isdigit() for char in token.text) or token.text.isupper():
                        technical_terms.append(token.text.lower())
            
            extracted_skills["technical_terms"] = set(technical_terms)
            
            # Extract named entities that might be skills
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]:
                    extracted_skills["entities"].add(ent.text.lower())
        
        # Convert sets to lists
        return {k: sorted(list(v)) for k, v in extracted_skills.items()}
    
    def analyze_experience_level(self, text: str) -> Dict[str, Any]:
        """Analyze experience level from resume text."""
        experience_indicators = {
            "entry": [
                r'\b(entry.level|junior|beginner|fresh.graduate|recent.graduate|intern)\b',
                r'\b(0.?-?.?[12].?years?|less.than.2.years?)\b'
            ],
            "mid": [
                r'\b(mid.level|intermediate|[3-5].?years?|experienced)\b',
                r'\b(some.experience|moderate.experience)\b'
            ],
            "senior": [
                r'\b(senior|sr\.|lead|principal|[6-9].?years?|extensive.experience)\b',
                r'\b(team.lead|technical.lead)\b'
            ],
            "expert": [
                r'\b(expert|architect|director|10\+?.?years?|principal.engineer)\b',
                r'\b(thought.leader|subject.matter.expert|sme)\b'
            ]
        }
        
        text_lower = text.lower()
        scores = defaultdict(int)
        
        for level, patterns in experience_indicators.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                scores[level] += len(matches)
        
        # Determine most likely level
        if scores:
            predicted_level = max(scores, key=scores.get)
        else:
            predicted_level = "unknown"
        
        return {
            "predicted_level": predicted_level,
            "confidence_scores": dict(scores),
            "indicators_found": sum(scores.values())
        }
    
    def calculate_resume_quality_score(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive resume quality metrics."""
        quality_metrics = {}
        
        # Length metrics
        word_count = len(text.split())
        quality_metrics["word_count"] = word_count
        quality_metrics["length_score"] = min(100, (word_count / 500) * 100)  # Optimal ~500 words
        
        # Readability metrics
        try:
            quality_metrics["readability"] = {
                "flesch_reading_ease": flesch_reading_ease(text),
                "flesch_kincaid_grade": flesch_kincaid_grade(text)
            }
        except:
            quality_metrics["readability"] = {"error": "Could not calculate readability"}
        
        # Structure analysis
        sections = self._identify_resume_sections(text)
        quality_metrics["sections_found"] = list(sections.keys())
        quality_metrics["structure_score"] = (len(sections) / 7) * 100  # Expect ~7 sections
        
        # Keyword density
        skills = self.extract_comprehensive_skills(text)
        total_skills = sum(len(v) for v in skills.values())
        quality_metrics["skill_density"] = (total_skills / word_count) * 100 if word_count > 0 else 0
        
        # Action verb usage
        action_verbs = self._count_action_verbs(text)
        quality_metrics["action_verbs"] = action_verbs
        quality_metrics["action_verb_score"] = min(100, (action_verbs / 20) * 100)  # Expect ~20 action verbs
        
        # Calculate overall score
        overall_score = np.mean([
            quality_metrics["length_score"],
            quality_metrics["structure_score"],
            min(100, quality_metrics["skill_density"] * 10),  # Cap at 100
            quality_metrics["action_verb_score"]
        ])
        
        quality_metrics["overall_score"] = round(overall_score, 2)
        
        # Recommendations
        recommendations = []
        if word_count < 300:
            recommendations.append("Resume is too short. Add more detail about your experience.")
        elif word_count > 800:
            recommendations.append("Resume may be too long. Consider condensing to 1-2 pages.")
        
        if len(sections) < 5:
            recommendations.append("Add more sections (e.g., Skills, Education, Experience, Summary).")
        
        if quality_metrics["skill_density"] < 2:
            recommendations.append("Include more technical skills and keywords.")
        
        if action_verbs < 10:
            recommendations.append("Use more action verbs to describe your achievements.")
        
        quality_metrics["recommendations"] = recommendations
        
        return quality_metrics
    
    def _identify_resume_sections(self, text: str) -> Dict[str, str]:
        """Identify standard resume sections."""
        section_patterns = {
            "summary": r'(summary|objective|profile|about)\s*:?\s*\n',
            "experience": r'(experience|employment|work\s+history)\s*:?\s*\n',
            "education": r'(education|academic|qualification)\s*:?\s*\n',
            "skills": r'(skills|technical\s+skills|competencies)\s*:?\s*\n',
            "projects": r'(projects|portfolio)\s*:?\s*\n',
            "certifications": r'(certifications?|licenses?)\s*:?\s*\n',
            "achievements": r'(achievements?|accomplishments?|awards?)\s*:?\s*\n'
        }
        
        sections = {}
        for section, pattern in section_patterns.items():
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                sections[section] = "found"
        
        return sections
    
    def _count_action_verbs(self, text: str) -> int:
        """Count action verbs in resume."""
        action_verbs = [
            'achieved', 'analyzed', 'built', 'created', 'designed', 'developed',
            'established', 'implemented', 'improved', 'increased', 'launched',
            'led', 'managed', 'optimized', 'organized', 'performed', 'planned',
            'produced', 'programmed', 'reduced', 'resolved', 'streamlined',
            'supervised', 'trained', 'transformed'
        ]
        
        text_lower = text.lower()
        count = 0
        for verb in action_verbs:
            count += len(re.findall(r'\b' + verb + r'\b', text_lower))
        
        return count


def load_deep_learning_models():
    """Load deep learning models for semantic analysis."""
    global sentence_model, bert_model, bert_tokenizer
    
    try:
        # Load sentence transformer for semantic similarity
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded sentence transformer model")
    except Exception as e:
        logger.warning(f"Could not load sentence transformer: {e}")
    
    try:
        # Load BERT for advanced text analysis
        bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        bert_model = AutoModel.from_pretrained('bert-base-uncased')
        logger.info("Loaded BERT model")
    except Exception as e:
        logger.warning(f"Could not load BERT model: {e}")


@mcp.tool()
async def analyze_resume_advanced(
    resume_text: str,
    analysis_depth: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Perform advanced analysis on resume using NLP and ML techniques.
    
    Args:
        resume_text: The resume text to analyze
        analysis_depth: Level of analysis (basic, moderate, comprehensive)
    
    Returns:
        Comprehensive analysis results
    """
    analyzer = ResumeAnalyzer()
    results = {"timestamp": datetime.now().isoformat()}
    
    # Basic analysis
    results["skills"] = analyzer.extract_comprehensive_skills(resume_text)
    results["experience_level"] = analyzer.analyze_experience_level(resume_text)
    results["quality_metrics"] = analyzer.calculate_resume_quality_score(resume_text)
    
    if analysis_depth in ["moderate", "comprehensive"]:
        # Extract contact info
        from kaggle_resume_server import ResumeProcessor
        processor = ResumeProcessor()
        results["contact_info"] = await extract_contact_info(resume_text)
        
        # Semantic analysis if models are loaded
        if sentence_model and analysis_depth == "comprehensive":
            # Extract key sentences
            sentences = [s.strip() for s in resume_text.split('.') if len(s.strip()) > 20]
            if sentences:
                # Encode sentences
                embeddings = sentence_model.encode(sentences[:20])  # Limit to 20 sentences
                
                # Find most important sentences (closest to centroid)
                centroid = np.mean(embeddings, axis=0)
                similarities = cosine_similarity([centroid], embeddings)[0]
                top_indices = np.argsort(similarities)[-5:][::-1]
                
                results["key_sentences"] = [
                    {
                        "sentence": sentences[i],
                        "importance_score": float(similarities[i])
                    }
                    for i in top_indices
                ]
    
    return results


@mcp.tool()
async def compare_resume_to_job(
    resume_text: str,
    job_description: str,
    match_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Compare resume to job description using semantic similarity.
    
    Args:
        resume_text: The resume text
        job_description: The job description text
        match_threshold: Minimum similarity score to consider a match
    
    Returns:
        Detailed comparison results
    """
    analyzer = ResumeAnalyzer()
    
    # Extract skills from both
    resume_skills = analyzer.extract_comprehensive_skills(resume_text)
    job_skills = analyzer.extract_comprehensive_skills(job_description)
    
    # Flatten skills
    resume_skill_set = set()
    for skills in resume_skills.values():
        resume_skill_set.update(skills)
    
    job_skill_set = set()
    for skills in job_skills.values():
        job_skill_set.update(skills)
    
    # Calculate skill overlap
    common_skills = resume_skill_set.intersection(job_skill_set)
    missing_skills = job_skill_set - resume_skill_set
    extra_skills = resume_skill_set - job_skill_set
    
    skill_match_score = len(common_skills) / len(job_skill_set) if job_skill_set else 0
    
    results = {
        "skill_analysis": {
            "common_skills": sorted(list(common_skills)),
            "missing_skills": sorted(list(missing_skills)),
            "extra_skills": sorted(list(extra_skills)),
            "skill_match_score": round(skill_match_score, 3)
        }
    }
    
    # Semantic similarity if model is available
    if sentence_model:
        # Encode full texts
        resume_embedding = sentence_model.encode([resume_text])
        job_embedding = sentence_model.encode([job_description])
        
        # Calculate similarity
        similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        results["semantic_similarity"] = float(similarity)
        
        # Extract and compare key phrases
        resume_sentences = [s.strip() for s in resume_text.split('.') if len(s.strip()) > 20][:10]
        job_sentences = [s.strip() for s in job_description.split('.') if len(s.strip()) > 20][:10]
        
        if resume_sentences and job_sentences:
            resume_embeddings = sentence_model.encode(resume_sentences)
            job_embeddings = sentence_model.encode(job_sentences)
            
            # Find best matching sentences
            cross_similarity = cosine_similarity(resume_embeddings, job_embeddings)
            best_matches = []
            
            for i, job_sent in enumerate(job_sentences):
                best_resume_idx = np.argmax(cross_similarity[:, i])
                best_score = cross_similarity[best_resume_idx, i]
                
                if best_score >= match_threshold:
                    best_matches.append({
                        "job_requirement": job_sent,
                        "resume_match": resume_sentences[best_resume_idx],
                        "similarity_score": float(best_score)
                    })
            
            results["sentence_matches"] = best_matches
    
    # Calculate overall match score
    overall_score = skill_match_score
    if "semantic_similarity" in results:
        overall_score = (skill_match_score + results["semantic_similarity"]) / 2
    
    results["overall_match_score"] = round(overall_score, 3)
    results["recommendation"] = "Strong match" if overall_score >= 0.7 else "Moderate match" if overall_score >= 0.5 else "Weak match"
    
    # Generate improvement suggestions
    suggestions = []
    if missing_skills:
        suggestions.append(f"Consider highlighting these skills if you have them: {', '.join(list(missing_skills)[:5])}")
    
    if overall_score < 0.7:
        suggestions.append("Tailor your resume more closely to the job description")
    
    results["improvement_suggestions"] = suggestions
    
    return results


@mcp.tool()
async def generate_skill_recommendations(
    resume_text: str,
    target_role: str,
    num_recommendations: int = 10
) -> Dict[str, Any]:
    """
    Generate skill recommendations based on resume and target role.
    
    Args:
        resume_text: The resume text
        target_role: Target job role/title
        num_recommendations: Number of recommendations to generate
    
    Returns:
        Skill recommendations with learning paths
    """
    analyzer = ResumeAnalyzer()
    
    # Extract current skills
    current_skills = analyzer.extract_comprehensive_skills(resume_text)
    all_current_skills = set()
    for skills in current_skills.values():
        all_current_skills.update(skills)
    
    # Define role-based skill requirements
    role_skills = {
        "data scientist": [
            "python", "r", "sql", "machine learning", "deep learning",
            "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn",
            "statistics", "data visualization", "tableau", "power bi"
        ],
        "full stack developer": [
            "javascript", "react", "node.js", "python", "django",
            "postgresql", "mongodb", "docker", "kubernetes", "aws",
            "rest api", "graphql", "git", "ci/cd"
        ],
        "devops engineer": [
            "docker", "kubernetes", "aws", "terraform", "ansible",
            "jenkins", "python", "bash", "linux", "monitoring",
            "prometheus", "grafana", "elk stack", "ci/cd"
        ],
        "machine learning engineer": [
            "python", "tensorflow", "pytorch", "mlflow", "kubeflow",
            "docker", "kubernetes", "aws sagemaker", "model deployment",
            "feature engineering", "a/b testing", "sql", "spark"
        ]
    }
    
    # Get target role skills
    target_role_lower = target_role.lower()
    target_skills = set()
    
    for role, skills in role_skills.items():
        if role in target_role_lower:
            target_skills.update(skills)
            break
    
    # If no exact match, use semantic similarity
    if not target_skills:
        # Use a default set based on common roles
        target_skills = set(role_skills.get("full stack developer", []))
    
    # Find skill gaps
    skill_gaps = target_skills - all_current_skills
    
    # Generate recommendations
    recommendations = []
    skill_ontology = analyzer.skill_ontology
    
    for skill in list(skill_gaps)[:num_recommendations]:
        related_skills = skill_ontology.get_related_skills(skill, depth=1)
        
        # Check if user has any related skills
        user_related = all_current_skills.intersection(related_skills)
        
        difficulty = "Beginner" if not user_related else "Intermediate"
        
        recommendation = {
            "skill": skill,
            "priority": "High" if skill in list(skill_gaps)[:3] else "Medium",
            "difficulty": difficulty,
            "related_skills_you_have": sorted(list(user_related)),
            "learning_resources": {
                "online_courses": f"Search for '{skill}' courses on Coursera, Udemy, or edX",
                "documentation": f"Official {skill} documentation",
                "projects": f"Build projects using {skill}"
            },
            "estimated_time": "2-3 months" if difficulty == "Beginner" else "1-2 months"
        }
        
        recommendations.append(recommendation)
    
    # Create learning path
    learning_path = []
    
    # Group by difficulty
    beginner_skills = [r for r in recommendations if r["difficulty"] == "Beginner"]
    intermediate_skills = [r for r in recommendations if r["difficulty"] == "Intermediate"]
    
    if beginner_skills:
        learning_path.append({
            "phase": "Foundation",
            "duration": "3-4 months",
            "skills": [r["skill"] for r in beginner_skills],
            "focus": "Build fundamental knowledge"
        })
    
    if intermediate_skills:
        learning_path.append({
            "phase": "Advanced",
            "duration": "2-3 months",
            "skills": [r["skill"] for r in intermediate_skills],
            "focus": "Deepen expertise and practical application"
        })
    
    return {
        "current_skill_count": len(all_current_skills),
        "target_role": target_role,
        "skill_gap_count": len(skill_gaps),
        "recommendations": recommendations,
        "learning_path": learning_path,
        "estimated_total_time": "6-12 months for significant role transition"
    }


@mcp.tool()
async def extract_achievements(
    resume_text: str,
    categorize: bool = True
) -> Dict[str, Any]:
    """
    Extract and analyze achievements from resume.
    
    Args:
        resume_text: The resume text
        categorize: Whether to categorize achievements
    
    Returns:
        Extracted achievements with metrics
    """
    # Patterns for achievements with metrics
    achievement_patterns = [
        r'(?:increased|improved|boosted|enhanced).{0,50}by\s+(\d+)\s*%',
        r'(?:reduced|decreased|cut|lowered).{0,50}by\s+(\d+)\s*%',
        r'(?:saved|generated|earned)\s+\$([\d,]+)',
        r'(?:led|managed|supervised)\s+(?:a\s+)?team\s+of\s+(\d+)',
        r'(?:delivered|completed|launched)\s+(\d+)\s+(?:projects?|features?|products?)',
        r'(?:achieved|exceeded|surpassed).{0,50}(?:targets?|goals?|objectives?)\s+by\s+(\d+)\s*%',
        r'(?:processed|handled|managed)\s+(\d+\+?)\s+(?:requests?|transactions?|cases?)'
    ]
    
    achievements = []
    
    # Extract sentences with achievement patterns
    sentences = [s.strip() for s in re.split(r'[.!?]', resume_text) if s.strip()]
    
    for sentence in sentences:
        for pattern in achievement_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                metric_value = match.group(1)
                achievements.append({
                    "text": sentence,
                    "metric": metric_value,
                    "type": "quantified"
                })
                break
    
    # Look for non-quantified achievements
    achievement_keywords = [
        'awarded', 'recognized', 'promoted', 'selected', 'chosen',
        'published', 'presented', 'certified', 'graduated', 'honors'
    ]
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in achievement_keywords):
            # Skip if already captured as quantified
            if not any(a["text"] == sentence for a in achievements):
                achievements.append({
                    "text": sentence,
                    "metric": None,
                    "type": "qualitative"
                })
    
    # Categorize achievements
    if categorize and achievements:
        categories = {
            "performance": [],
            "leadership": [],
            "innovation": [],
            "recognition": [],
            "education": []
        }
        
        for achievement in achievements:
            text_lower = achievement["text"].lower()
            
            if any(word in text_lower for word in ['increased', 'improved', 'exceeded', 'surpassed']):
                categories["performance"].append(achievement)
            elif any(word in text_lower for word in ['led', 'managed', 'supervised', 'mentored']):
                categories["leadership"].append(achievement)
            elif any(word in text_lower for word in ['created', 'developed', 'designed', 'implemented']):
                categories["innovation"].append(achievement)
            elif any(word in text_lower for word in ['awarded', 'recognized', 'honored', 'certified']):
                categories["recognition"].append(achievement)
            elif any(word in text_lower for word in ['graduated', 'gpa', 'degree', 'certification']):
                categories["education"].append(achievement)
        
        return {
            "total_achievements": len(achievements),
            "quantified_achievements": len([a for a in achievements if a["type"] == "quantified"]),
            "categories": {k: v for k, v in categories.items() if v},
            "all_achievements": achievements
        }
    
    return {
        "total_achievements": len(achievements),
        "achievements": achievements
    }


@mcp.tool()
async def optimize_resume_keywords(
    resume_text: str,
    job_descriptions: List[str],
    industry: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optimize resume keywords based on job descriptions.
    
    Args:
        resume_text: The resume text
        job_descriptions: List of target job descriptions
        industry: Optional industry specification
    
    Returns:
        Keyword optimization recommendations
    """
    analyzer = ResumeAnalyzer()
    
    # Extract current keywords from resume
    resume_skills = analyzer.extract_comprehensive_skills(resume_text)
    resume_keywords = set()
    for skills in resume_skills.values():
        resume_keywords.update(skills)
    
    # Extract keywords from all job descriptions
    job_keywords = Counter()
    all_job_skills = set()
    
    for job_desc in job_descriptions:
        job_skills = analyzer.extract_comprehensive_skills(job_desc)
        for skill_type, skills in job_skills.items():
            for skill in skills:
                job_keywords[skill] += 1
                all_job_skills.add(skill)
    
    # Find most common keywords across job descriptions
    common_keywords = [keyword for keyword, count in job_keywords.most_common(20)]
    
    # Calculate keyword coverage
    covered_keywords = resume_keywords.intersection(all_job_skills)
    missing_keywords = all_job_skills - resume_keywords
    
    coverage_score = len(covered_keywords) / len(all_job_skills) if all_job_skills else 0
    
    # Industry-specific keywords
    industry_keywords = {
        "technology": ["agile", "scrum", "ci/cd", "cloud", "devops", "api", "microservices"],
        "finance": ["financial analysis", "risk management", "compliance", "trading", "portfolio"],
        "healthcare": ["hipaa", "ehr", "clinical", "patient care", "medical"],
        "marketing": ["seo", "sem", "analytics", "campaign", "roi", "conversion"],
        "data": ["analytics", "visualization", "etl", "data pipeline", "bi", "reporting"]
    }
    
    recommended_industry_keywords = []
    if industry and industry.lower() in industry_keywords:
        industry_specific = industry_keywords[industry.lower()]
        recommended_industry_keywords = [k for k in industry_specific if k not in resume_keywords]
    
    # Generate ATS optimization tips
    ats_tips = []
    
    if coverage_score < 0.5:
        ats_tips.append("Your resume has low keyword coverage. Add more relevant keywords.")
    
    if len(missing_keywords) > 10:
        top_missing = sorted(missing_keywords, key=lambda x: job_keywords.get(x, 0), reverse=True)[:5]
        ats_tips.append(f"Consider adding these high-frequency keywords: {', '.join(top_missing)}")
    
    if not any(keyword in resume_text.lower() for keyword in ['results', 'achieved', 'improved']):
        ats_tips.append("Use more result-oriented language with quantifiable achievements.")
    
    # Keyword density analysis
    resume_words = resume_text.lower().split()
    keyword_density = {}
    
    for keyword in common_keywords[:10]:
        count = resume_words.count(keyword)
        density = (count / len(resume_words)) * 100 if resume_words else 0
        keyword_density[keyword] = {
            "count": count,
            "density": round(density, 2)
        }
    
    return {
        "keyword_coverage_score": round(coverage_score, 3),
        "total_relevant_keywords": len(all_job_skills),
        "keywords_in_resume": len(covered_keywords),
        "missing_keywords": sorted(list(missing_keywords))[:15],
        "most_common_job_keywords": common_keywords,
        "keyword_density": keyword_density,
        "industry_recommendations": recommended_industry_keywords,
        "ats_optimization_tips": ats_tips,
        "optimization_priority": "High" if coverage_score < 0.5 else "Medium" if coverage_score < 0.7 else "Low"
    }


# Initialize server on startup
@mcp.on_startup()
async def startup():
    """Initialize models on server startup."""
    logger.info("Advanced Resume Analyzer MCP Server starting...")
    
    # Load deep learning models in background
    asyncio.create_task(asyncio.to_thread(load_deep_learning_models))
    
    logger.info("Server ready to accept requests.")


# Run the server
if __name__ == "__main__":
    mcp.run()
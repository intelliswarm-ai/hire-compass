#!/usr/bin/env python3
"""
Example usage of the Kaggle Resume MCP servers.

This script demonstrates how to use both the Kaggle Resume Server
and Advanced Resume Analyzer for various resume analysis tasks.
"""

import asyncio
import json
from typing import Dict, Any, List
from fastmcp import FastMCP


class ResumeMCPClient:
    """Client for interacting with Resume MCP servers."""
    
    def __init__(self, kaggle_url: str = "http://localhost:8000", 
                 advanced_url: str = "http://localhost:8001"):
        self.kaggle_client = FastMCP(kaggle_url)
        self.advanced_client = FastMCP(advanced_url)
    
    async def analyze_resume_basic(self, resume_text: str) -> Dict[str, Any]:
        """Basic resume analysis using Kaggle server."""
        result = await self.kaggle_client.call_tool(
            "analyze_resume",
            resume_text=resume_text,
            include_skills=True,
            include_category=True,
            include_experience=True,
            include_education=True
        )
        return result
    
    async def analyze_resume_advanced(self, resume_text: str) -> Dict[str, Any]:
        """Advanced resume analysis with NLP."""
        result = await self.advanced_client.call_tool(
            "analyze_resume_advanced",
            resume_text=resume_text,
            analysis_depth="comprehensive"
        )
        return result
    
    async def match_resume_to_job(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """Match resume to job description."""
        result = await self.advanced_client.call_tool(
            "compare_resume_to_job",
            resume_text=resume_text,
            job_description=job_description,
            match_threshold=0.7
        )
        return result
    
    async def find_similar_candidates(self, resume_text: str, dataset_path: str) -> List[Dict[str, Any]]:
        """Find similar resumes from dataset."""
        result = await self.kaggle_client.call_tool(
            "find_similar_resumes",
            resume_text=resume_text,
            dataset_path=dataset_path,
            top_k=5,
            min_similarity=0.6
        )
        return result
    
    async def get_skill_recommendations(self, resume_text: str, target_role: str) -> Dict[str, Any]:
        """Get personalized skill recommendations."""
        result = await self.advanced_client.call_tool(
            "generate_skill_recommendations",
            resume_text=resume_text,
            target_role=target_role,
            num_recommendations=10
        )
        return result


async def example_complete_analysis():
    """Example: Complete resume analysis workflow."""
    client = ResumeMCPClient()
    
    # Sample resume text
    resume_text = """
    John Doe
    john.doe@email.com | +1-234-567-8900 | linkedin.com/in/johndoe | github.com/johndoe
    
    SUMMARY
    Experienced Python Developer with 5+ years building scalable web applications and data pipelines.
    Expertise in Django, FastAPI, and cloud technologies. Led team of 4 developers to deliver 
    enterprise solutions that increased efficiency by 40%.
    
    TECHNICAL SKILLS
    Languages: Python, JavaScript, SQL, Java
    Frameworks: Django, FastAPI, React, Flask
    Databases: PostgreSQL, MongoDB, Redis
    Cloud: AWS (EC2, S3, Lambda), Docker, Kubernetes
    Tools: Git, Jenkins, JIRA, Confluence
    
    EXPERIENCE
    Senior Python Developer | TechCorp Inc. | 2020 - Present
    - Developed microservices architecture serving 1M+ daily users
    - Reduced API response time by 60% through optimization
    - Mentored junior developers and conducted code reviews
    - Implemented CI/CD pipeline reducing deployment time by 75%
    
    Python Developer | StartupXYZ | 2018 - 2020
    - Built RESTful APIs using Django REST Framework
    - Integrated third-party services and payment gateways
    - Improved test coverage from 45% to 85%
    - Collaborated with cross-functional teams in Agile environment
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology | 2014 - 2018
    GPA: 3.8/4.0
    
    CERTIFICATIONS
    - AWS Certified Solutions Architect
    - Python Institute Certified Professional
    """
    
    print("=== COMPLETE RESUME ANALYSIS ===\n")
    
    # 1. Basic Analysis
    print("1. Basic Resume Analysis")
    print("-" * 50)
    basic_analysis = await client.analyze_resume_basic(resume_text)
    
    print(f"Skills Found: {basic_analysis.get('skill_count', 0)}")
    print(f"Category Prediction: {basic_analysis.get('ensemble_prediction', {}).get('category', 'Unknown')}")
    print(f"Experience Years: {basic_analysis.get('experience_years', 'Not found')}")
    print(f"Education: {', '.join(basic_analysis.get('education', []))}")
    print()
    
    # 2. Advanced Analysis
    print("2. Advanced Resume Analysis")
    print("-" * 50)
    advanced_analysis = await client.analyze_resume_advanced(resume_text)
    
    quality_metrics = advanced_analysis.get('quality_metrics', {})
    print(f"Overall Quality Score: {quality_metrics.get('overall_score', 0):.2f}/100")
    print(f"Word Count: {quality_metrics.get('word_count', 0)}")
    print(f"Action Verbs Used: {quality_metrics.get('action_verbs', 0)}")
    print("\nRecommendations:")
    for rec in quality_metrics.get('recommendations', []):
        print(f"  - {rec}")
    print()
    
    # 3. Job Matching
    print("3. Job Description Matching")
    print("-" * 50)
    job_description = """
    We are looking for a Senior Python Developer to join our team.
    
    Requirements:
    - 5+ years of Python development experience
    - Strong experience with Django or FastAPI
    - Knowledge of microservices architecture
    - Experience with AWS services
    - PostgreSQL and NoSQL databases
    - Docker and Kubernetes experience
    - Strong problem-solving skills
    - Team leadership experience
    """
    
    match_result = await client.match_resume_to_job(resume_text, job_description)
    print(f"Overall Match Score: {match_result.get('overall_match_score', 0):.2%}")
    print(f"Skill Match Score: {match_result.get('skill_analysis', {}).get('skill_match_score', 0):.2%}")
    print(f"\nCommon Skills: {', '.join(match_result.get('skill_analysis', {}).get('common_skills', [])[:5])}")
    print(f"Missing Skills: {', '.join(match_result.get('skill_analysis', {}).get('missing_skills', [])[:5])}")
    print()
    
    # 4. Skill Recommendations
    print("4. Skill Recommendations for Data Science Role")
    print("-" * 50)
    recommendations = await client.get_skill_recommendations(resume_text, "Data Scientist")
    
    print(f"Current Skills: {recommendations.get('current_skill_count', 0)}")
    print(f"Skill Gap: {recommendations.get('skill_gap_count', 0)} skills")
    print("\nTop 5 Recommended Skills:")
    for i, rec in enumerate(recommendations.get('recommendations', [])[:5], 1):
        print(f"  {i}. {rec['skill']} (Priority: {rec['priority']}, Difficulty: {rec['difficulty']})")
    print()
    
    # 5. Extract Achievements
    print("5. Achievement Extraction")
    print("-" * 50)
    achievements = await client.advanced_client.call_tool(
        "extract_achievements",
        resume_text=resume_text,
        categorize=True
    )
    
    print(f"Total Achievements Found: {achievements.get('total_achievements', 0)}")
    print(f"Quantified Achievements: {achievements.get('quantified_achievements', 0)}")
    
    categories = achievements.get('categories', {})
    for category, items in categories.items():
        if items:
            print(f"\n{category.title()} Achievements:")
            for achievement in items[:2]:  # Show first 2 from each category
                print(f"  - {achievement['text'][:80]}...")


async def example_batch_processing():
    """Example: Batch processing multiple resumes."""
    client = ResumeMCPClient()
    
    # Sample resumes
    resumes = [
        {
            "id": "resume_001",
            "text": "Software Engineer with Python and Java experience..."
        },
        {
            "id": "resume_002", 
            "text": "Data Scientist specializing in machine learning..."
        },
        {
            "id": "resume_003",
            "text": "DevOps Engineer with AWS and Kubernetes expertise..."
        }
    ]
    
    print("=== BATCH RESUME PROCESSING ===\n")
    
    result = await client.kaggle_client.call_tool(
        "batch_analyze_resumes",
        resumes=resumes,
        target_category="Python Developer"
    )
    
    print(f"Total Analyzed: {result.get('total_analyzed', 0)}")
    print(f"Matches Found: {result.get('matches', 0)}")
    print("\nCategory Distribution:")
    for category, count in result.get('category_distribution', {}).items():
        print(f"  {category}: {count}")


async def example_ats_optimization():
    """Example: ATS keyword optimization."""
    client = ResumeMCPClient()
    
    resume_text = "Python developer with experience in web development..."
    
    job_descriptions = [
        "Senior Python Developer needed with Django, AWS, and microservices experience...",
        "Full Stack Developer - Python, React, PostgreSQL, Docker required...",
        "Python Engineer - FastAPI, Kubernetes, CI/CD pipeline experience..."
    ]
    
    print("=== ATS KEYWORD OPTIMIZATION ===\n")
    
    result = await client.advanced_client.call_tool(
        "optimize_resume_keywords",
        resume_text=resume_text,
        job_descriptions=job_descriptions,
        industry="technology"
    )
    
    print(f"Keyword Coverage Score: {result.get('keyword_coverage_score', 0):.2%}")
    print(f"Missing Keywords: {', '.join(result.get('missing_keywords', [])[:10])}")
    print("\nATS Optimization Tips:")
    for tip in result.get('ats_optimization_tips', []):
        print(f"  - {tip}")


async def main():
    """Run all examples."""
    print("Resume MCP Server Examples")
    print("=" * 60)
    print()
    
    try:
        # Run complete analysis example
        await example_complete_analysis()
        print("\n" + "=" * 60 + "\n")
        
        # Run batch processing example
        await example_batch_processing()
        print("\n" + "=" * 60 + "\n")
        
        # Run ATS optimization example
        await example_ats_optimization()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the MCP servers are running:")
        print("  python mcp_server/kaggle_resume_server.py")
        print("  python mcp_server/advanced_resume_analyzer.py")


if __name__ == "__main__":
    asyncio.run(main())
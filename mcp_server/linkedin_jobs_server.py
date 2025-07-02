#!/usr/bin/env python3
"""
LinkedIn Jobs MCP Server.

This server provides tools for fetching job positions from LinkedIn
and matching them against resumes. Uses web scraping with proper
rate limiting and caching to be respectful of LinkedIn's servers.

Note: This implementation uses public job listings only and respects
LinkedIn's robots.txt and terms of service.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urljoin, urlparse

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from fake_useragent import UserAgent
import aiofiles
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("linkedin-jobs-analyzer")

# Global cache for jobs
jobs_cache = {}
cache_expiry = {}

# Rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 2.0  # seconds between requests


class JobLevel(Enum):
    """Job experience levels."""
    INTERNSHIP = "Internship"
    ENTRY_LEVEL = "Entry level"
    ASSOCIATE = "Associate"
    MID_SENIOR = "Mid-Senior level"
    DIRECTOR = "Director"
    EXECUTIVE = "Executive"


class JobType(Enum):
    """Job types."""
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"
    CONTRACT = "Contract"
    TEMPORARY = "Temporary"
    VOLUNTEER = "Volunteer"
    INTERNSHIP = "Internship"


@dataclass
class LinkedInJob:
    """LinkedIn job posting data."""
    job_id: str
    title: str
    company: str
    location: str
    posted_date: Optional[str] = None
    description: Optional[str] = None
    seniority_level: Optional[str] = None
    employment_type: Optional[str] = None
    job_function: Optional[str] = None
    industries: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    salary_range: Optional[Dict[str, Any]] = None
    apply_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "posted_date": self.posted_date,
            "description": self.description,
            "seniority_level": self.seniority_level,
            "employment_type": self.employment_type,
            "job_function": self.job_function,
            "industries": self.industries,
            "requirements": self.requirements,
            "benefits": self.benefits,
            "salary_range": self.salary_range,
            "apply_url": self.apply_url
        }


class LinkedInJobScraper:
    """Scraper for LinkedIn public job listings."""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session = None
        self.headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    async def __aenter__(self):
        """Create aiohttp session."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
    
    async def search_company_jobs(
        self,
        company_name: str,
        location: Optional[str] = None,
        job_type: Optional[str] = None,
        experience_level: Optional[str] = None,
        limit: int = 25
    ) -> List[LinkedInJob]:
        """
        Search for jobs from a specific company.
        
        Note: This is a simplified implementation. In production, you would
        need to handle LinkedIn's anti-scraping measures more robustly.
        """
        jobs = []
        
        # Build search URL
        base_url = "https://www.linkedin.com/jobs/search/"
        params = {
            'keywords': f'"{company_name}"',
            'location': location or '',
            'f_C': company_name,  # Company filter
        }
        
        if job_type:
            params['f_JT'] = job_type
        if experience_level:
            params['f_E'] = experience_level
        
        # Rate limiting
        await self._rate_limit()
        
        try:
            # Make request
            async with self.session.get(base_url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch jobs: {response.status}")
                    return jobs
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Parse job listings
                job_cards = soup.find_all('div', class_='base-card')[:limit]
                
                for card in job_cards:
                    job = self._parse_job_card(card)
                    if job:
                        jobs.append(job)
        
        except Exception as e:
            logger.error(f"Error scraping LinkedIn jobs: {e}")
        
        return jobs
    
    def _parse_job_card(self, card) -> Optional[LinkedInJob]:
        """Parse job information from a job card."""
        try:
            # Extract basic info
            title_elem = card.find('h3', class_='base-search-card__title')
            company_elem = card.find('h4', class_='base-search-card__subtitle')
            location_elem = card.find('span', class_='job-search-card__location')
            link_elem = card.find('a', class_='base-card__full-link')
            
            if not all([title_elem, company_elem, link_elem]):
                return None
            
            # Extract job ID from URL
            job_url = link_elem.get('href', '')
            job_id = self._extract_job_id(job_url)
            
            job = LinkedInJob(
                job_id=job_id,
                title=title_elem.text.strip(),
                company=company_elem.text.strip(),
                location=location_elem.text.strip() if location_elem else '',
                apply_url=job_url
            )
            
            # Extract posted date
            time_elem = card.find('time')
            if time_elem:
                job.posted_date = time_elem.get('datetime', time_elem.text.strip())
            
            return job
            
        except Exception as e:
            logger.error(f"Error parsing job card: {e}")
            return None
    
    def _extract_job_id(self, url: str) -> str:
        """Extract job ID from LinkedIn URL."""
        # LinkedIn job URLs typically contain the job ID
        match = re.search(r'/view/(\d+)', url)
        if match:
            return match.group(1)
        return url.split('/')[-1].split('?')[0]
    
    async def get_job_details(self, job_url: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed job information."""
        await self._rate_limit()
        
        try:
            async with self.session.get(job_url) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                details = {}
                
                # Extract description
                desc_elem = soup.find('div', class_='show-more-less-html__markup')
                if desc_elem:
                    details['description'] = desc_elem.text.strip()
                    details['requirements'] = self._extract_requirements(desc_elem.text)
                
                # Extract job criteria
                criteria_container = soup.find('ul', class_='description__job-criteria-list')
                if criteria_container:
                    criteria_items = criteria_container.find_all('li')
                    for item in criteria_items:
                        label = item.find('h3')
                        value = item.find('span')
                        if label and value:
                            label_text = label.text.strip().lower()
                            if 'seniority' in label_text:
                                details['seniority_level'] = value.text.strip()
                            elif 'employment' in label_text:
                                details['employment_type'] = value.text.strip()
                            elif 'function' in label_text:
                                details['job_function'] = value.text.strip()
                            elif 'industries' in label_text:
                                details['industries'] = [i.strip() for i in value.text.split(',')]
                
                return details
                
        except Exception as e:
            logger.error(f"Error fetching job details: {e}")
            return None
    
    def _extract_requirements(self, description: str) -> List[str]:
        """Extract requirements from job description."""
        requirements = []
        
        # Common requirement patterns
        patterns = [
            r'(?:requirements?|qualifications?|you have|you.ll need|must have)[:\n]*(.*?)(?:\n\n|$)',
            r'(?:required skills?|technical skills?)[:\n]*(.*?)(?:\n\n|$)',
            r'(?:\d+\+?\s*years?\s+(?:of\s+)?experience\s+(?:in|with)\s+[\w\s,]+)',
            r'(?:bachelor|master|phd|degree)(?:.s)?\s+(?:in\s+)?[\w\s,]+',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, description, re.IGNORECASE | re.DOTALL)
            for match in matches:
                text = match.group(0).strip()
                if len(text) > 10 and len(text) < 500:  # Reasonable length
                    requirements.append(text)
        
        # Also look for bullet points
        bullet_pattern = r'[•·▪-]\s*(.+?)(?:\n|$)'
        bullets = re.findall(bullet_pattern, description)
        requirements.extend([b.strip() for b in bullets if len(b.strip()) > 10])
        
        # Deduplicate
        seen = set()
        unique_requirements = []
        for req in requirements:
            req_lower = req.lower()
            if req_lower not in seen:
                seen.add(req_lower)
                unique_requirements.append(req)
        
        return unique_requirements[:10]  # Limit to top 10
    
    async def _rate_limit(self):
        """Implement rate limiting."""
        global last_request_time
        current_time = time.time()
        time_since_last = current_time - last_request_time
        
        if time_since_last < MIN_REQUEST_INTERVAL:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - time_since_last)
        
        last_request_time = time.time()


class JobMatcher:
    """Match jobs with resumes."""
    
    def __init__(self):
        self.skill_weights = {
            'exact_match': 1.0,
            'similar_match': 0.7,
            'category_match': 0.5
        }
    
    def calculate_match_score(
        self,
        resume_data: Dict[str, Any],
        job: LinkedInJob
    ) -> Dict[str, Any]:
        """Calculate match score between resume and job."""
        scores = {}
        
        # Extract resume components
        resume_skills = set(resume_data.get('skills', []))
        resume_experience = resume_data.get('experience_years', 0)
        resume_text = resume_data.get('text', '').lower()
        
        # Title match
        title_score = self._calculate_title_match(
            resume_data.get('current_title', ''),
            job.title
        )
        scores['title_match'] = title_score
        
        # Skills match
        job_requirements = ' '.join(job.requirements).lower()
        job_description = (job.description or '').lower()
        
        skills_score = self._calculate_skills_match(
            resume_skills,
            job_requirements + ' ' + job_description
        )
        scores['skills_match'] = skills_score
        
        # Experience match
        exp_score = self._calculate_experience_match(
            resume_experience,
            job.seniority_level
        )
        scores['experience_match'] = exp_score
        
        # Location match
        location_score = self._calculate_location_match(
            resume_data.get('location', ''),
            job.location
        )
        scores['location_match'] = location_score
        
        # Calculate overall score
        weights = {
            'title_match': 0.25,
            'skills_match': 0.40,
            'experience_match': 0.25,
            'location_match': 0.10
        }
        
        overall_score = sum(
            scores[key] * weights[key]
            for key in weights
            if key in scores
        )
        
        return {
            'overall_score': round(overall_score, 3),
            'scores': scores,
            'match_level': self._get_match_level(overall_score),
            'missing_skills': self._identify_missing_skills(
                resume_skills,
                job_requirements
            )
        }
    
    def _calculate_title_match(self, resume_title: str, job_title: str) -> float:
        """Calculate title similarity."""
        if not resume_title or not job_title:
            return 0.0
        
        resume_words = set(resume_title.lower().split())
        job_words = set(job_title.lower().split())
        
        # Remove common words
        common_words = {'senior', 'junior', 'sr', 'jr', 'lead', 'principal'}
        resume_words -= common_words
        job_words -= common_words
        
        if not resume_words or not job_words:
            return 0.0
        
        intersection = resume_words.intersection(job_words)
        union = resume_words.union(job_words)
        
        return len(intersection) / len(union)
    
    def _calculate_skills_match(
        self,
        resume_skills: Set[str],
        job_text: str
    ) -> float:
        """Calculate skills match score."""
        if not resume_skills:
            return 0.0
        
        matched_skills = 0
        for skill in resume_skills:
            if skill.lower() in job_text:
                matched_skills += 1
        
        return min(1.0, matched_skills / max(len(resume_skills), 5))
    
    def _calculate_experience_match(
        self,
        resume_years: float,
        seniority_level: Optional[str]
    ) -> float:
        """Calculate experience level match."""
        if not seniority_level:
            return 0.5  # Neutral score
        
        level_ranges = {
            'Internship': (0, 1),
            'Entry level': (0, 3),
            'Associate': (2, 5),
            'Mid-Senior level': (4, 8),
            'Director': (8, 15),
            'Executive': (10, None)
        }
        
        if seniority_level not in level_ranges:
            return 0.5
        
        min_years, max_years = level_ranges[seniority_level]
        
        if min_years <= resume_years:
            if max_years is None or resume_years <= max_years:
                return 1.0
            else:
                # Over-qualified
                return 0.7
        else:
            # Under-qualified
            gap = min_years - resume_years
            return max(0.3, 1.0 - (gap * 0.2))
    
    def _calculate_location_match(
        self,
        resume_location: str,
        job_location: str
    ) -> float:
        """Calculate location match."""
        if not resume_location or not job_location:
            return 0.5
        
        # Simple check - in production, use geocoding
        resume_loc_lower = resume_location.lower()
        job_loc_lower = job_location.lower()
        
        # Check for remote
        if 'remote' in job_loc_lower:
            return 1.0
        
        # Check for city/state match
        resume_parts = set(resume_loc_lower.split(','))
        job_parts = set(job_loc_lower.split(','))
        
        if resume_parts.intersection(job_parts):
            return 1.0
        
        return 0.0
    
    def _get_match_level(self, score: float) -> str:
        """Determine match level from score."""
        if score >= 0.8:
            return "Excellent Match"
        elif score >= 0.6:
            return "Good Match"
        elif score >= 0.4:
            return "Fair Match"
        else:
            return "Poor Match"
    
    def _identify_missing_skills(
        self,
        resume_skills: Set[str],
        job_requirements: str
    ) -> List[str]:
        """Identify skills mentioned in job but not in resume."""
        # Common technical skills to look for
        technical_skills = {
            'python', 'java', 'javascript', 'react', 'angular', 'vue',
            'django', 'flask', 'spring', 'nodejs', 'aws', 'azure', 'gcp',
            'docker', 'kubernetes', 'sql', 'nosql', 'mongodb', 'postgresql',
            'machine learning', 'deep learning', 'data science', 'analytics',
            'agile', 'scrum', 'git', 'ci/cd', 'devops', 'microservices'
        }
        
        missing = []
        job_lower = job_requirements.lower()
        
        for skill in technical_skills:
            if skill in job_lower and skill not in {s.lower() for s in resume_skills}:
                missing.append(skill)
        
        return missing[:5]  # Top 5 missing skills


@mcp.tool()
async def fetch_company_jobs(
    company_name: str,
    location: Optional[str] = None,
    job_type: Optional[str] = None,
    experience_level: Optional[str] = None,
    limit: int = 25,
    use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch job positions from a specific company on LinkedIn.
    
    Args:
        company_name: Name of the company
        location: Job location filter
        job_type: Type of job (Full-time, Part-time, etc.)
        experience_level: Experience level filter
        limit: Maximum number of jobs to fetch
        use_cache: Whether to use cached results
    
    Returns:
        List of job postings
    """
    cache_key = f"{company_name}_{location}_{job_type}_{experience_level}"
    
    # Check cache
    if use_cache and cache_key in jobs_cache:
        cache_time = cache_expiry.get(cache_key, 0)
        if time.time() < cache_time:
            logger.info(f"Returning cached results for {company_name}")
            return jobs_cache[cache_key]
    
    # Fetch fresh data
    async with LinkedInJobScraper() as scraper:
        jobs = await scraper.search_company_jobs(
            company_name=company_name,
            location=location,
            job_type=job_type,
            experience_level=experience_level,
            limit=limit
        )
    
    # Convert to dict format
    job_dicts = [job.to_dict() for job in jobs]
    
    # Cache results
    jobs_cache[cache_key] = job_dicts
    cache_expiry[cache_key] = time.time() + 3600  # 1 hour cache
    
    return job_dicts


@mcp.tool()
async def match_resume_to_company_jobs(
    resume_text: str,
    company_name: str,
    resume_data: Optional[Dict[str, Any]] = None,
    location: Optional[str] = None,
    min_match_score: float = 0.5
) -> Dict[str, Any]:
    """
    Match a resume against all jobs from a specific company.
    
    Args:
        resume_text: The resume text
        company_name: Name of the company
        resume_data: Pre-processed resume data (skills, experience, etc.)
        location: Location preference
        min_match_score: Minimum score to include in results
    
    Returns:
        Matching results with scores and recommendations
    """
    # Fetch company jobs
    jobs = await fetch_company_jobs(
        company_name=company_name,
        location=location,
        limit=50
    )
    
    if not jobs:
        return {
            "error": f"No jobs found for {company_name}",
            "company": company_name,
            "total_jobs": 0
        }
    
    # Process resume if data not provided
    if not resume_data:
        # Import from other server
        from advanced_resume_analyzer import ResumeAnalyzer
        analyzer = ResumeAnalyzer()
        
        skills = analyzer.extract_comprehensive_skills(resume_text)
        all_skills = []
        for skill_list in skills.values():
            all_skills.extend(skill_list)
        
        resume_data = {
            "text": resume_text,
            "skills": all_skills,
            "experience_years": analyzer.analyze_experience_level(resume_text).get(
                "predicted_level", "unknown"
            )
        }
    
    # Match against each job
    matcher = JobMatcher()
    matches = []
    
    for job_dict in jobs:
        job = LinkedInJob(**job_dict)
        match_result = matcher.calculate_match_score(resume_data, job)
        
        if match_result['overall_score'] >= min_match_score:
            matches.append({
                "job": job_dict,
                "match": match_result
            })
    
    # Sort by match score
    matches.sort(key=lambda x: x['match']['overall_score'], reverse=True)
    
    # Calculate statistics
    score_distribution = {
        "excellent": len([m for m in matches if m['match']['overall_score'] >= 0.8]),
        "good": len([m for m in matches if 0.6 <= m['match']['overall_score'] < 0.8]),
        "fair": len([m for m in matches if 0.4 <= m['match']['overall_score'] < 0.6]),
        "poor": len([m for m in matches if m['match']['overall_score'] < 0.4])
    }
    
    # Get top missing skills across all jobs
    all_missing_skills = []
    for match in matches[:10]:  # Top 10 matches
        all_missing_skills.extend(match['match'].get('missing_skills', []))
    
    skill_frequency = {}
    for skill in all_missing_skills:
        skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
    
    top_missing_skills = sorted(
        skill_frequency.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    return {
        "company": company_name,
        "total_jobs": len(jobs),
        "matched_jobs": len(matches),
        "match_rate": round(len(matches) / len(jobs), 2) if jobs else 0,
        "score_distribution": score_distribution,
        "top_matches": matches[:10],
        "top_missing_skills": [skill for skill, _ in top_missing_skills],
        "recommendations": _generate_recommendations(matches, resume_data)
    }


@mcp.tool()
async def analyze_company_job_trends(
    company_name: str,
    location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze job posting trends for a company.
    
    Args:
        company_name: Name of the company
        location: Location filter
    
    Returns:
        Analysis of job trends and requirements
    """
    # Fetch jobs
    jobs = await fetch_company_jobs(
        company_name=company_name,
        location=location,
        limit=100
    )
    
    if not jobs:
        return {"error": f"No jobs found for {company_name}"}
    
    # Analyze trends
    analysis = {
        "company": company_name,
        "total_jobs": len(jobs),
        "locations": {},
        "job_types": {},
        "seniority_levels": {},
        "common_requirements": [],
        "top_skills": []
    }
    
    # Location distribution
    for job in jobs:
        loc = job.get('location', 'Unknown')
        analysis['locations'][loc] = analysis['locations'].get(loc, 0) + 1
    
    # Job type distribution
    for job in jobs:
        jtype = job.get('employment_type', 'Unknown')
        analysis['job_types'][jtype] = analysis['job_types'].get(jtype, 0) + 1
    
    # Seniority distribution
    for job in jobs:
        level = job.get('seniority_level', 'Unknown')
        analysis['seniority_levels'][level] = analysis['seniority_levels'].get(level, 0) + 1
    
    # Extract common requirements
    all_requirements = []
    for job in jobs:
        all_requirements.extend(job.get('requirements', []))
    
    # Simple frequency analysis
    req_frequency = {}
    for req in all_requirements:
        req_lower = req.lower()
        # Extract key phrases
        if 'years' in req_lower:
            req_frequency[req] = req_frequency.get(req, 0) + 1
    
    analysis['common_requirements'] = sorted(
        req_frequency.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Extract top skills
    skill_keywords = {
        'python', 'java', 'javascript', 'react', 'angular', 'aws',
        'docker', 'kubernetes', 'sql', 'machine learning', 'agile',
        'data science', 'devops', 'cloud', 'api', 'microservices'
    }
    
    skill_counts = {}
    for job in jobs:
        job_text = ' '.join([
            job.get('title', ''),
            job.get('description', ''),
            ' '.join(job.get('requirements', []))
        ]).lower()
        
        for skill in skill_keywords:
            if skill in job_text:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    analysis['top_skills'] = sorted(
        skill_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]
    
    return analysis


@mcp.tool()
async def get_job_details(
    job_url: str,
    extract_requirements: bool = True
) -> Dict[str, Any]:
    """
    Get detailed information about a specific job.
    
    Args:
        job_url: LinkedIn job URL
        extract_requirements: Whether to extract detailed requirements
    
    Returns:
        Detailed job information
    """
    async with LinkedInJobScraper() as scraper:
        details = await scraper.get_job_details(job_url)
    
    if not details:
        return {"error": "Could not fetch job details"}
    
    return details


@mcp.tool()
async def bulk_match_resume(
    resume_text: str,
    company_list: List[str],
    location: Optional[str] = None,
    min_match_score: float = 0.6
) -> Dict[str, Any]:
    """
    Match resume against multiple companies.
    
    Args:
        resume_text: The resume text
        company_list: List of company names
        location: Location preference
        min_match_score: Minimum match score
    
    Returns:
        Matching results across all companies
    """
    # Process resume once
    from advanced_resume_analyzer import ResumeAnalyzer
    analyzer = ResumeAnalyzer()
    
    skills = analyzer.extract_comprehensive_skills(resume_text)
    all_skills = []
    for skill_list in skills.values():
        all_skills.extend(skill_list)
    
    resume_data = {
        "text": resume_text,
        "skills": all_skills,
        "experience_years": analyzer.analyze_experience_level(resume_text).get(
            "predicted_level", "unknown"
        )
    }
    
    # Match against each company
    company_results = []
    total_matches = 0
    all_job_matches = []
    
    for company in company_list:
        try:
            result = await match_resume_to_company_jobs(
                resume_text=resume_text,
                company_name=company,
                resume_data=resume_data,
                location=location,
                min_match_score=min_match_score
            )
            
            if 'error' not in result:
                company_results.append({
                    "company": company,
                    "total_jobs": result['total_jobs'],
                    "matched_jobs": result['matched_jobs'],
                    "match_rate": result['match_rate'],
                    "top_match_score": result['top_matches'][0]['match']['overall_score'] 
                        if result['top_matches'] else 0
                })
                
                total_matches += result['matched_jobs']
                
                # Add top matches to global list
                for match in result['top_matches'][:3]:
                    match['company'] = company
                    all_job_matches.append(match)
        
        except Exception as e:
            logger.error(f"Error processing company {company}: {e}")
            company_results.append({
                "company": company,
                "error": str(e)
            })
    
    # Sort all matches by score
    all_job_matches.sort(key=lambda x: x['match']['overall_score'], reverse=True)
    
    # Calculate aggregate statistics
    successful_companies = [r for r in company_results if 'error' not in r]
    
    return {
        "companies_analyzed": len(company_list),
        "companies_with_matches": len([r for r in successful_companies if r['matched_jobs'] > 0]),
        "total_jobs_found": sum(r['total_jobs'] for r in successful_companies),
        "total_matches": total_matches,
        "company_results": company_results,
        "top_matches_overall": all_job_matches[:10],
        "average_match_rate": round(
            sum(r['match_rate'] for r in successful_companies) / len(successful_companies), 2
        ) if successful_companies else 0
    }


def _generate_recommendations(matches: List[Dict], resume_data: Dict) -> List[str]:
    """Generate recommendations based on matching results."""
    recommendations = []
    
    if not matches:
        recommendations.append("No matching positions found. Consider broadening your search criteria.")
        return recommendations
    
    # Analyze match scores
    avg_score = sum(m['match']['overall_score'] for m in matches) / len(matches)
    
    if avg_score < 0.5:
        recommendations.append("Overall match scores are low. Consider:")
        recommendations.append("- Updating your skills to match current market demands")
        recommendations.append("- Gaining more experience in key areas")
    elif avg_score < 0.7:
        recommendations.append("You have moderate matches. To improve:")
        recommendations.append("- Focus on acquiring the most commonly required skills")
        recommendations.append("- Tailor your resume to highlight relevant experience")
    else:
        recommendations.append("You have strong matches! Consider:")
        recommendations.append("- Applying to the top-matched positions immediately")
        recommendations.append("- Networking with employees at these companies")
    
    # Skill recommendations
    all_missing = []
    for match in matches[:5]:
        all_missing.extend(match['match'].get('missing_skills', []))
    
    if all_missing:
        top_missing = list(set(all_missing))[:3]
        recommendations.append(f"- Prioritize learning: {', '.join(top_missing)}")
    
    return recommendations


# Server startup
@mcp.on_startup()
async def startup():
    """Initialize LinkedIn Jobs MCP Server."""
    logger.info("LinkedIn Jobs MCP Server starting...")
    logger.info("Note: This server respects LinkedIn's terms of service")
    logger.info("Server ready to accept requests.")


# Run the server
if __name__ == "__main__":
    mcp.run()
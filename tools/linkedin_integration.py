"""
LinkedIn Integration Tools.

This module provides tools for integrating with LinkedIn job data,
including both web scraping and API approaches. It connects with the
MCP servers for job fetching and matching.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import aiohttp
import pandas as pd
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from src.shared.protocols import Cache, Logger
from src.infrastructure.adapters.circuit_breaker import CircuitBreakerImpl
from src.infrastructure.adapters.rate_limiter import TokenBucketRateLimiter


logger = logging.getLogger(__name__)


class LinkedInJobSearchInput(BaseModel):
    """Input for LinkedIn job search."""
    company_name: str = Field(description="Name of the company")
    location: Optional[str] = Field(None, description="Job location")
    job_type: Optional[str] = Field(None, description="Type of job")
    limit: int = Field(25, description="Maximum number of jobs to fetch")


class LinkedInJobMatchInput(BaseModel):
    """Input for LinkedIn job matching."""
    resume_text: str = Field(description="Resume text to match")
    company_name: str = Field(description="Company name to match against")
    location: Optional[str] = Field(None, description="Preferred location")
    min_match_score: float = Field(0.5, description="Minimum match score")


class LinkedInJobSearchTool(BaseTool):
    """Tool for searching LinkedIn jobs."""
    
    name = "linkedin_job_search"
    description = "Search for job positions from a specific company on LinkedIn"
    args_schema = LinkedInJobSearchInput
    
    def __init__(
        self,
        mcp_server_url: str = "http://localhost:8002",
        cache: Optional[Cache] = None,
        logger: Optional[Logger] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mcp_server_url = mcp_server_url
        self.cache = cache
        self.logger = logger
        self.circuit_breaker = CircuitBreakerImpl(
            name="linkedin_search",
            failure_threshold=3,
            recovery_timeout=60
        )
        self.rate_limiter = TokenBucketRateLimiter(
            rate=10,  # 10 requests per second
            burst=20
        )
    
    async def _arun(
        self,
        company_name: str,
        location: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 25
    ) -> str:
        """Async implementation of job search."""
        # Check cache first
        cache_key = f"linkedin_jobs:{company_name}:{location}:{job_type}"
        if self.cache:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return json.dumps(cached_result, indent=2)
        
        # Rate limiting
        await self.rate_limiter.consume(cache_key)
        
        # Make request with circuit breaker
        async def fetch_jobs():
            async with aiohttp.ClientSession() as session:
                url = f"{self.mcp_server_url}/tools/fetch_company_jobs"
                payload = {
                    "company_name": company_name,
                    "location": location,
                    "job_type": job_type,
                    "limit": limit
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"MCP server error: {error_text}")
        
        try:
            result = await self.circuit_breaker.call(fetch_jobs)
            
            # Cache result
            if self.cache and result:
                await self.cache.set(cache_key, result, ttl=3600)  # 1 hour
            
            # Format response
            if isinstance(result, list):
                summary = {
                    "company": company_name,
                    "total_jobs": len(result),
                    "jobs": result[:10],  # First 10 jobs
                    "locations": list(set(job.get("location", "") for job in result)),
                    "job_types": list(set(job.get("employment_type", "") for job in result))
                }
                return json.dumps(summary, indent=2)
            else:
                return json.dumps(result, indent=2)
                
        except Exception as e:
            error_msg = f"Error searching LinkedIn jobs: {str(e)}"
            if self.logger:
                self.logger.error(error_msg, error=e)
            return error_msg
    
    def _run(self, *args, **kwargs) -> str:
        """Sync wrapper for async implementation."""
        return asyncio.run(self._arun(*args, **kwargs))


class LinkedInJobMatchTool(BaseTool):
    """Tool for matching resume with LinkedIn jobs."""
    
    name = "linkedin_job_match"
    description = "Match a resume against job positions from a company on LinkedIn"
    args_schema = LinkedInJobMatchInput
    
    def __init__(
        self,
        mcp_server_url: str = "http://localhost:8002",
        advanced_analyzer_url: str = "http://localhost:8001",
        cache: Optional[Cache] = None,
        logger: Optional[Logger] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mcp_server_url = mcp_server_url
        self.advanced_analyzer_url = advanced_analyzer_url
        self.cache = cache
        self.logger = logger
    
    async def _arun(
        self,
        resume_text: str,
        company_name: str,
        location: Optional[str] = None,
        min_match_score: float = 0.5
    ) -> str:
        """Async implementation of job matching."""
        try:
            # First, analyze the resume
            resume_analysis = await self._analyze_resume(resume_text)
            
            # Then, match against company jobs
            async with aiohttp.ClientSession() as session:
                url = f"{self.mcp_server_url}/tools/match_resume_to_company_jobs"
                payload = {
                    "resume_text": resume_text,
                    "company_name": company_name,
                    "resume_data": resume_analysis,
                    "location": location,
                    "min_match_score": min_match_score
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._format_match_results(result)
                    else:
                        error_text = await response.text()
                        raise Exception(f"MCP server error: {error_text}")
                        
        except Exception as e:
            error_msg = f"Error matching resume with LinkedIn jobs: {str(e)}"
            if self.logger:
                self.logger.error(error_msg, error=e)
            return error_msg
    
    async def _analyze_resume(self, resume_text: str) -> Dict[str, Any]:
        """Analyze resume using advanced analyzer."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.advanced_analyzer_url}/tools/analyze_resume_advanced"
                payload = {
                    "resume_text": resume_text,
                    "analysis_depth": "moderate"
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        analysis = await response.json()
                        
                        # Extract relevant data for matching
                        skills = []
                        for skill_type, skill_list in analysis.get("skills", {}).items():
                            skills.extend(skill_list)
                        
                        return {
                            "skills": skills,
                            "experience_years": analysis.get("experience_level", {}).get(
                                "predicted_level", "unknown"
                            ),
                            "current_title": self._extract_current_title(resume_text),
                            "location": analysis.get("contact_info", {}).get("location", "")
                        }
                    else:
                        return {"skills": [], "experience_years": 0}
                        
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Resume analysis failed: {e}")
            return {"skills": [], "experience_years": 0}
    
    def _extract_current_title(self, resume_text: str) -> str:
        """Extract current job title from resume."""
        lines = resume_text.split('\n')
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['developer', 'engineer', 'analyst', 'manager', 'designer']):
                return line.strip()
        return ""
    
    def _format_match_results(self, result: Dict[str, Any]) -> str:
        """Format matching results for display."""
        output = []
        output.append(f"Company: {result.get('company', 'Unknown')}")
        output.append(f"Total Jobs Found: {result.get('total_jobs', 0)}")
        output.append(f"Matched Jobs: {result.get('matched_jobs', 0)}")
        output.append(f"Match Rate: {result.get('match_rate', 0):.1%}")
        
        # Score distribution
        dist = result.get('score_distribution', {})
        output.append("\nMatch Distribution:")
        output.append(f"  Excellent: {dist.get('excellent', 0)}")
        output.append(f"  Good: {dist.get('good', 0)}")
        output.append(f"  Fair: {dist.get('fair', 0)}")
        output.append(f"  Poor: {dist.get('poor', 0)}")
        
        # Top matches
        top_matches = result.get('top_matches', [])[:5]
        if top_matches:
            output.append("\nTop 5 Matches:")
            for i, match in enumerate(top_matches, 1):
                job = match['job']
                score = match['match']['overall_score']
                output.append(f"\n{i}. {job['title']} - {job['location']}")
                output.append(f"   Score: {score:.1%}")
                output.append(f"   Type: {job.get('employment_type', 'Not specified')}")
                output.append(f"   Level: {job.get('seniority_level', 'Not specified')}")
        
        # Recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            output.append("\nRecommendations:")
            for rec in recommendations:
                output.append(f"  • {rec}")
        
        return '\n'.join(output)
    
    def _run(self, *args, **kwargs) -> str:
        """Sync wrapper for async implementation."""
        return asyncio.run(self._arun(*args, **kwargs))


class LinkedInBulkAnalysisTool(BaseTool):
    """Tool for bulk analysis of LinkedIn jobs across multiple companies."""
    
    name = "linkedin_bulk_analysis"
    description = "Analyze job opportunities across multiple companies on LinkedIn"
    
    def __init__(
        self,
        mcp_server_url: str = "http://localhost:8002",
        cache: Optional[Cache] = None,
        logger: Optional[Logger] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mcp_server_url = mcp_server_url
        self.cache = cache
        self.logger = logger
    
    async def _arun(
        self,
        resume_text: str,
        company_list: List[str],
        location: Optional[str] = None,
        min_match_score: float = 0.6
    ) -> str:
        """Analyze resume against multiple companies."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.mcp_server_url}/tools/bulk_match_resume"
                payload = {
                    "resume_text": resume_text,
                    "company_list": company_list,
                    "location": location,
                    "min_match_score": min_match_score
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._format_bulk_results(result)
                    else:
                        error_text = await response.text()
                        raise Exception(f"MCP server error: {error_text}")
                        
        except Exception as e:
            error_msg = f"Error in bulk analysis: {str(e)}"
            if self.logger:
                self.logger.error(error_msg, error=e)
            return error_msg
    
    def _format_bulk_results(self, result: Dict[str, Any]) -> str:
        """Format bulk analysis results."""
        output = []
        output.append("=== LinkedIn Job Analysis Across Multiple Companies ===")
        output.append(f"\nCompanies Analyzed: {result.get('companies_analyzed', 0)}")
        output.append(f"Companies with Matches: {result.get('companies_with_matches', 0)}")
        output.append(f"Total Jobs Found: {result.get('total_jobs_found', 0)}")
        output.append(f"Total Matches: {result.get('total_matches', 0)}")
        output.append(f"Average Match Rate: {result.get('average_match_rate', 0):.1%}")
        
        # Company results
        company_results = result.get('company_results', [])
        if company_results:
            output.append("\nCompany Breakdown:")
            for comp in sorted(company_results, key=lambda x: x.get('top_match_score', 0), reverse=True):
                if 'error' not in comp:
                    output.append(f"\n  {comp['company']}:")
                    output.append(f"    Jobs: {comp['total_jobs']}")
                    output.append(f"    Matches: {comp['matched_jobs']}")
                    output.append(f"    Match Rate: {comp['match_rate']:.1%}")
                    output.append(f"    Best Score: {comp['top_match_score']:.1%}")
        
        # Top matches overall
        top_matches = result.get('top_matches_overall', [])[:5]
        if top_matches:
            output.append("\n\nTop 5 Matches Overall:")
            for i, match in enumerate(top_matches, 1):
                job = match['job']
                score = match['match']['overall_score']
                output.append(f"\n{i}. {job['title']} at {match.get('company', job['company'])}")
                output.append(f"   Location: {job['location']}")
                output.append(f"   Score: {score:.1%}")
        
        return '\n'.join(output)
    
    def _run(self, *args, **kwargs) -> str:
        """Sync wrapper for async implementation."""
        return asyncio.run(self._arun(*args, **kwargs))


class LinkedInCompanyAnalysisTool(BaseTool):
    """Tool for analyzing job trends at a specific company."""
    
    name = "linkedin_company_analysis"
    description = "Analyze job posting trends and requirements for a specific company"
    
    def __init__(
        self,
        mcp_server_url: str = "http://localhost:8002",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mcp_server_url = mcp_server_url
    
    async def _arun(
        self,
        company_name: str,
        location: Optional[str] = None
    ) -> str:
        """Analyze company job trends."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.mcp_server_url}/tools/analyze_company_job_trends"
                payload = {
                    "company_name": company_name,
                    "location": location
                }
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._format_analysis(result)
                    else:
                        error_text = await response.text()
                        raise Exception(f"MCP server error: {error_text}")
                        
        except Exception as e:
            return f"Error analyzing company: {str(e)}"
    
    def _format_analysis(self, result: Dict[str, Any]) -> str:
        """Format company analysis results."""
        output = []
        output.append(f"=== Job Analysis for {result.get('company', 'Unknown')} ===")
        output.append(f"\nTotal Open Positions: {result.get('total_jobs', 0)}")
        
        # Location distribution
        locations = result.get('locations', {})
        if locations:
            output.append("\nJob Locations:")
            for loc, count in sorted(locations.items(), key=lambda x: x[1], reverse=True)[:5]:
                output.append(f"  • {loc}: {count} positions")
        
        # Seniority levels
        levels = result.get('seniority_levels', {})
        if levels:
            output.append("\nSeniority Levels:")
            for level, count in sorted(levels.items(), key=lambda x: x[1], reverse=True):
                output.append(f"  • {level}: {count} positions")
        
        # Top skills
        skills = result.get('top_skills', [])
        if skills:
            output.append("\nMost Requested Skills:")
            for skill, count in skills[:10]:
                output.append(f"  • {skill}: {count} mentions")
        
        # Common requirements
        reqs = result.get('common_requirements', [])
        if reqs:
            output.append("\nCommon Requirements:")
            for req, count in reqs[:5]:
                output.append(f"  • {req[:100]}...")
        
        return '\n'.join(output)
    
    def _run(self, *args, **kwargs) -> str:
        """Sync wrapper for async implementation."""
        return asyncio.run(self._arun(*args, **kwargs))


# Factory function to create LinkedIn tools
def create_linkedin_tools(
    mcp_server_url: str = "http://localhost:8002",
    advanced_analyzer_url: str = "http://localhost:8001",
    cache: Optional[Cache] = None,
    logger: Optional[Logger] = None
) -> List[BaseTool]:
    """Create all LinkedIn integration tools."""
    return [
        LinkedInJobSearchTool(
            mcp_server_url=mcp_server_url,
            cache=cache,
            logger=logger
        ),
        LinkedInJobMatchTool(
            mcp_server_url=mcp_server_url,
            advanced_analyzer_url=advanced_analyzer_url,
            cache=cache,
            logger=logger
        ),
        LinkedInBulkAnalysisTool(
            mcp_server_url=mcp_server_url,
            cache=cache,
            logger=logger
        ),
        LinkedInCompanyAnalysisTool(
            mcp_server_url=mcp_server_url
        )
    ]
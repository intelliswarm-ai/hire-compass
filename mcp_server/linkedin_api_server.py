#!/usr/bin/env python3
"""
LinkedIn API MCP Server (Official API Version).

This server provides tools for fetching job positions using LinkedIn's
official APIs and OAuth authentication. This is the recommended approach
for production use.

Requirements:
- LinkedIn Developer Account
- OAuth 2.0 credentials
- Appropriate API permissions
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlencode

import aiohttp
from fastmcp import FastMCP
from dataclasses import dataclass, field
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("linkedin-api-server")

# LinkedIn API configuration
LINKEDIN_API_BASE = "https://api.linkedin.com/v2"
LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"


@dataclass
class LinkedInCredentials:
    """LinkedIn API credentials."""
    client_id: str
    client_secret: str
    redirect_uri: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None


class LinkedInAPIClient:
    """Official LinkedIn API client."""
    
    def __init__(self, credentials: LinkedInCredentials):
        self.credentials = credentials
        self.session = None
    
    async def __aenter__(self):
        """Create aiohttp session."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
    
    def get_authorization_url(self, state: str, scope: List[str]) -> str:
        """Generate OAuth authorization URL."""
        params = {
            'response_type': 'code',
            'client_id': self.credentials.client_id,
            'redirect_uri': self.credentials.redirect_uri,
            'state': state,
            'scope': ' '.join(scope)
        }
        return f"{LINKEDIN_AUTH_URL}?{urlencode(params)}"
    
    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': self.credentials.redirect_uri,
            'client_id': self.credentials.client_id,
            'client_secret': self.credentials.client_secret
        }
        
        async with self.session.post(LINKEDIN_TOKEN_URL, data=data) as response:
            if response.status == 200:
                token_data = await response.json()
                self.credentials.access_token = token_data['access_token']
                self.credentials.token_expires_at = datetime.now() + timedelta(
                    seconds=token_data['expires_in']
                )
                return token_data
            else:
                error = await response.text()
                raise Exception(f"Token exchange failed: {error}")
    
    async def _ensure_valid_token(self):
        """Ensure we have a valid access token."""
        if not self.credentials.access_token:
            raise Exception("No access token available. Please authenticate first.")
        
        if self.credentials.token_expires_at and datetime.now() >= self.credentials.token_expires_at:
            raise Exception("Access token expired. Please re-authenticate.")
    
    async def _make_api_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make authenticated API request."""
        await self._ensure_valid_token()
        
        headers = {
            'Authorization': f'Bearer {self.credentials.access_token}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0'
        }
        
        url = f"{LINKEDIN_API_BASE}{endpoint}"
        
        async with self.session.request(
            method,
            url,
            headers=headers,
            params=params,
            json=data
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error = await response.text()
                raise Exception(f"API request failed: {error}")
    
    async def search_jobs(
        self,
        keywords: Optional[str] = None,
        location: Optional[str] = None,
        company_ids: Optional[List[str]] = None,
        job_type: Optional[str] = None,
        experience_level: Optional[str] = None,
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Search for jobs using LinkedIn API.
        
        Note: Requires appropriate API permissions.
        """
        params = {
            'q': 'jobs',
            'start': 0,
            'count': min(limit, 50)  # LinkedIn API limit
        }
        
        if keywords:
            params['keywords'] = keywords
        if location:
            params['locationId'] = location
        if company_ids:
            params['companyIds'] = ','.join(company_ids)
        
        try:
            response = await self._make_api_request('GET', '/jobs', params=params)
            return response.get('elements', [])
        except Exception as e:
            logger.error(f"Job search failed: {e}")
            return []
    
    async def get_company_info(self, company_id: str) -> Dict[str, Any]:
        """Get company information."""
        try:
            response = await self._make_api_request(
                'GET',
                f'/organizations/{company_id}'
            )
            return response
        except Exception as e:
            logger.error(f"Failed to get company info: {e}")
            return {}
    
    async def get_job_posting(self, job_id: str) -> Dict[str, Any]:
        """Get detailed job posting information."""
        try:
            response = await self._make_api_request(
                'GET',
                f'/jobs/{job_id}'
            )
            return response
        except Exception as e:
            logger.error(f"Failed to get job posting: {e}")
            return {}


class LinkedInJobProcessor:
    """Process LinkedIn job data."""
    
    @staticmethod
    def process_job_posting(raw_job: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw LinkedIn API job data."""
        processed = {
            'job_id': raw_job.get('id'),
            'title': raw_job.get('title'),
            'company': raw_job.get('companyName'),
            'location': raw_job.get('locationName'),
            'posted_date': LinkedInJobProcessor._parse_date(
                raw_job.get('listedAt')
            ),
            'description': raw_job.get('description'),
            'employment_type': raw_job.get('employmentType'),
            'seniority_level': raw_job.get('seniorityLevel'),
            'industries': raw_job.get('industries', []),
            'job_functions': raw_job.get('jobFunctions', []),
            'apply_url': raw_job.get('applyUrl')
        }
        
        # Extract requirements
        if processed['description']:
            processed['requirements'] = LinkedInJobProcessor._extract_requirements(
                processed['description']
            )
        
        return processed
    
    @staticmethod
    def _parse_date(timestamp: Optional[int]) -> Optional[str]:
        """Parse LinkedIn timestamp."""
        if timestamp:
            return datetime.fromtimestamp(timestamp / 1000).isoformat()
        return None
    
    @staticmethod
    def _extract_requirements(description: str) -> List[str]:
        """Extract requirements from job description."""
        import re
        
        requirements = []
        
        # Look for requirements section
        req_pattern = r'(?:requirements?|qualifications?)[:\n]+(.*?)(?:\n\n|$)'
        match = re.search(req_pattern, description, re.IGNORECASE | re.DOTALL)
        
        if match:
            req_text = match.group(1)
            # Split by bullet points or newlines
            lines = re.split(r'[•·▪\-\n]+', req_text)
            requirements = [
                line.strip() for line in lines
                if line.strip() and len(line.strip()) > 10
            ]
        
        return requirements[:10]


# Global credentials storage
credentials_store = {}


@mcp.tool()
async def setup_linkedin_auth(
    client_id: str,
    client_secret: str,
    redirect_uri: str
) -> Dict[str, Any]:
    """
    Set up LinkedIn OAuth credentials.
    
    Args:
        client_id: LinkedIn app client ID
        client_secret: LinkedIn app client secret
        redirect_uri: OAuth redirect URI
    
    Returns:
        Authorization URL and session info
    """
    import secrets
    
    # Generate state for security
    state = secrets.token_urlsafe(32)
    
    # Create credentials
    creds = LinkedInCredentials(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri
    )
    
    # Store credentials
    credentials_store[state] = creds
    
    # Generate auth URL
    client = LinkedInAPIClient(creds)
    auth_url = client.get_authorization_url(
        state=state,
        scope=['r_liteprofile', 'r_jobs', 'w_member_social']
    )
    
    return {
        'auth_url': auth_url,
        'state': state,
        'instructions': 'Visit the auth_url to authorize the application'
    }


@mcp.tool()
async def complete_linkedin_auth(
    state: str,
    code: str
) -> Dict[str, Any]:
    """
    Complete LinkedIn OAuth flow.
    
    Args:
        state: OAuth state parameter
        code: Authorization code from LinkedIn
    
    Returns:
        Authentication status
    """
    if state not in credentials_store:
        return {'error': 'Invalid state parameter'}
    
    creds = credentials_store[state]
    
    async with LinkedInAPIClient(creds) as client:
        try:
            token_data = await client.exchange_code_for_token(code)
            return {
                'status': 'success',
                'expires_in': token_data['expires_in'],
                'message': 'Authentication successful'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


@mcp.tool()
async def search_linkedin_jobs(
    keywords: Optional[str] = None,
    company_name: Optional[str] = None,
    location: Optional[str] = None,
    job_type: Optional[str] = None,
    experience_level: Optional[str] = None,
    limit: int = 25,
    auth_state: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for jobs using LinkedIn API.
    
    Args:
        keywords: Search keywords
        company_name: Company name filter
        location: Location filter
        job_type: Type of job
        experience_level: Experience level
        limit: Maximum results
        auth_state: Authentication state
    
    Returns:
        List of job postings
    """
    if not auth_state or auth_state not in credentials_store:
        return [{
            'error': 'Authentication required',
            'message': 'Please complete OAuth flow first'
        }]
    
    creds = credentials_store[auth_state]
    
    async with LinkedInAPIClient(creds) as client:
        try:
            # If company name provided, first get company ID
            company_ids = None
            if company_name:
                # This would require company search API
                # For now, we'll use keywords
                keywords = f"{keywords} {company_name}" if keywords else company_name
            
            # Search jobs
            raw_jobs = await client.search_jobs(
                keywords=keywords,
                location=location,
                company_ids=company_ids,
                job_type=job_type,
                experience_level=experience_level,
                limit=limit
            )
            
            # Process results
            processor = LinkedInJobProcessor()
            processed_jobs = []
            
            for raw_job in raw_jobs:
                processed = processor.process_job_posting(raw_job)
                processed_jobs.append(processed)
            
            return processed_jobs
            
        except Exception as e:
            logger.error(f"Job search failed: {e}")
            return [{
                'error': str(e),
                'message': 'Failed to search jobs'
            }]


@mcp.tool()
async def get_linkedin_job_details(
    job_id: str,
    auth_state: str
) -> Dict[str, Any]:
    """
    Get detailed information about a specific job.
    
    Args:
        job_id: LinkedIn job ID
        auth_state: Authentication state
    
    Returns:
        Detailed job information
    """
    if auth_state not in credentials_store:
        return {
            'error': 'Authentication required'
        }
    
    creds = credentials_store[auth_state]
    
    async with LinkedInAPIClient(creds) as client:
        try:
            raw_job = await client.get_job_posting(job_id)
            processor = LinkedInJobProcessor()
            return processor.process_job_posting(raw_job)
        except Exception as e:
            return {
                'error': str(e)
            }


@mcp.tool()
async def match_resume_to_linkedin_jobs(
    resume_text: str,
    company_name: Optional[str] = None,
    keywords: Optional[str] = None,
    location: Optional[str] = None,
    auth_state: Optional[str] = None,
    min_match_score: float = 0.5
) -> Dict[str, Any]:
    """
    Match resume against LinkedIn job postings.
    
    Args:
        resume_text: Resume content
        company_name: Company filter
        keywords: Search keywords
        location: Location filter
        auth_state: Authentication state
        min_match_score: Minimum match score
    
    Returns:
        Matching results
    """
    # Search for jobs
    jobs = await search_linkedin_jobs(
        keywords=keywords,
        company_name=company_name,
        location=location,
        auth_state=auth_state,
        limit=50
    )
    
    if not jobs or 'error' in jobs[0]:
        return {
            'error': 'Failed to fetch jobs',
            'details': jobs[0] if jobs else 'No jobs found'
        }
    
    # Import matcher from other server
    from linkedin_jobs_server import JobMatcher
    from advanced_resume_analyzer import ResumeAnalyzer
    
    # Process resume
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
    
    # Match against jobs
    matcher = JobMatcher()
    matches = []
    
    for job in jobs:
        # Convert to expected format
        from linkedin_jobs_server import LinkedInJob
        job_obj = LinkedInJob(
            job_id=job.get('job_id', ''),
            title=job.get('title', ''),
            company=job.get('company', ''),
            location=job.get('location', ''),
            description=job.get('description', ''),
            seniority_level=job.get('seniority_level'),
            employment_type=job.get('employment_type'),
            requirements=job.get('requirements', [])
        )
        
        match_result = matcher.calculate_match_score(resume_data, job_obj)
        
        if match_result['overall_score'] >= min_match_score:
            matches.append({
                "job": job,
                "match": match_result
            })
    
    # Sort by score
    matches.sort(key=lambda x: x['match']['overall_score'], reverse=True)
    
    return {
        "total_jobs": len(jobs),
        "matched_jobs": len(matches),
        "top_matches": matches[:10],
        "average_match_score": round(
            sum(m['match']['overall_score'] for m in matches) / len(matches), 2
        ) if matches else 0
    }


# Server startup
@mcp.on_startup()
async def startup():
    """Initialize LinkedIn API MCP Server."""
    logger.info("LinkedIn API MCP Server starting...")
    logger.info("This server uses LinkedIn's official APIs")
    logger.info("Please ensure you have proper API credentials")
    logger.info("Server ready to accept requests.")


# Run the server
if __name__ == "__main__":
    mcp.run()
"""
Async web scraper for salary research with improved performance.
"""

import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import time
import re
import logging
from urllib.parse import quote_plus
from fake_useragent import UserAgent
from config import settings

logger = logging.getLogger(__name__)


class AsyncSalaryWebScraper:
    """Async web scraper for salary research"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.rate_limit_delay = 1.0  # Delay between requests to be respectful
        self._last_request_time = {}
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Create aiohttp session with proper headers"""
        return aiohttp.ClientSession(
            headers={
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            },
            timeout=self.timeout
        )
    
    async def _rate_limit(self, domain: str):
        """Implement rate limiting per domain"""
        current_time = time.time()
        if domain in self._last_request_time:
            elapsed = current_time - self._last_request_time[domain]
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time[domain] = time.time()
    
    async def search_glassdoor_salaries(self, job_title: str, location: str) -> Dict[str, Any]:
        """Search Glassdoor for salary information (mock implementation)"""
        try:
            # Note: In production, you would need proper API access or respect robots.txt
            # This is a simplified example structure
            await self._rate_limit("glassdoor.com")
            
            # Mock data for demonstration
            # In real implementation, you would scrape or use API
            return {
                "source": "Glassdoor",
                "title": job_title,
                "location": location,
                "salary_range": {
                    "min": 80000,
                    "max": 120000,
                    "average": 100000
                },
                "data_points": 150,
                "last_updated": "2024-01-15"
            }
            
        except Exception as e:
            logger.error(f"Error searching Glassdoor: {e}")
            return {}
    
    async def search_indeed_salaries(self, job_title: str, location: str) -> Dict[str, Any]:
        """Search Indeed for salary information"""
        try:
            await self._rate_limit("indeed.com")
            
            # Mock implementation
            return {
                "source": "Indeed",
                "title": job_title,
                "location": location,
                "salary_range": {
                    "min": 75000,
                    "max": 115000,
                    "average": 95000
                },
                "data_points": 200,
                "last_updated": "2024-01-10"
            }
            
        except Exception as e:
            logger.error(f"Error searching Indeed: {e}")
            return {}
    
    async def search_payscale_salaries(self, job_title: str, location: str, 
                                      experience_years: int = None) -> Dict[str, Any]:
        """Search PayScale for salary information"""
        try:
            await self._rate_limit("payscale.com")
            
            # Mock implementation
            base_salary = 90000
            if experience_years:
                # Adjust based on experience
                if experience_years < 3:
                    multiplier = 0.8
                elif experience_years < 7:
                    multiplier = 1.0
                elif experience_years < 12:
                    multiplier = 1.2
                else:
                    multiplier = 1.4
                
                base_salary = int(base_salary * multiplier)
            
            return {
                "source": "PayScale",
                "title": job_title,
                "location": location,
                "experience_years": experience_years,
                "salary_range": {
                    "min": int(base_salary * 0.85),
                    "max": int(base_salary * 1.15),
                    "average": base_salary
                },
                "percentiles": {
                    "10th": int(base_salary * 0.75),
                    "25th": int(base_salary * 0.85),
                    "50th": base_salary,
                    "75th": int(base_salary * 1.15),
                    "90th": int(base_salary * 1.30)
                },
                "last_updated": "2024-01-12"
            }
            
        except Exception as e:
            logger.error(f"Error searching PayScale: {e}")
            return {}
    
    async def search_linkedin_salaries(self, job_title: str, location: str,
                                     company: str = None) -> Dict[str, Any]:
        """Search LinkedIn for salary insights"""
        try:
            await self._rate_limit("linkedin.com")
            
            # Mock implementation
            base_salary = 95000
            if company:
                # Premium companies typically pay more
                if company.lower() in ['google', 'facebook', 'amazon', 'apple', 'microsoft']:
                    base_salary = int(base_salary * 1.3)
            
            return {
                "source": "LinkedIn",
                "title": job_title,
                "location": location,
                "company": company,
                "salary_range": {
                    "min": int(base_salary * 0.9),
                    "max": int(base_salary * 1.1),
                    "average": base_salary
                },
                "insights": {
                    "total_compensation": int(base_salary * 1.2),
                    "base_salary": base_salary,
                    "additional_compensation": int(base_salary * 0.2)
                },
                "last_updated": "2024-01-14"
            }
            
        except Exception as e:
            logger.error(f"Error searching LinkedIn: {e}")
            return {}
    
    async def aggregate_salary_data(self, job_title: str, location: str,
                                  experience_years: int = None,
                                  company: str = None) -> Dict[str, Any]:
        """Aggregate salary data from multiple sources concurrently"""
        try:
            # Run all searches concurrently
            tasks = [
                self.search_glassdoor_salaries(job_title, location),
                self.search_indeed_salaries(job_title, location),
                self.search_payscale_salaries(job_title, location, experience_years),
                self.search_linkedin_salaries(job_title, location, company)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out errors and empty results
            valid_results = []
            for result in results:
                if isinstance(result, dict) and result:
                    valid_results.append(result)
            
            if not valid_results:
                return {
                    "error": "No salary data found",
                    "sources_checked": 4,
                    "successful_sources": 0
                }
            
            # Calculate aggregated data
            all_averages = []
            all_mins = []
            all_maxs = []
            
            for result in valid_results:
                if "salary_range" in result:
                    salary_range = result["salary_range"]
                    if "average" in salary_range:
                        all_averages.append(salary_range["average"])
                    if "min" in salary_range:
                        all_mins.append(salary_range["min"])
                    if "max" in salary_range:
                        all_maxs.append(salary_range["max"])
            
            # Calculate aggregate statistics
            aggregate_data = {
                "job_title": job_title,
                "location": location,
                "experience_years": experience_years,
                "company": company,
                "aggregated_salary": {
                    "min": min(all_mins) if all_mins else None,
                    "max": max(all_maxs) if all_maxs else None,
                    "average": sum(all_averages) // len(all_averages) if all_averages else None,
                    "median": sorted(all_averages)[len(all_averages)//2] if all_averages else None
                },
                "sources": valid_results,
                "sources_count": len(valid_results),
                "confidence_score": len(valid_results) / 4.0  # Percentage of successful sources
            }
            
            return aggregate_data
            
        except Exception as e:
            logger.error(f"Error aggregating salary data: {e}")
            return {
                "error": str(e),
                "sources_checked": 4,
                "successful_sources": 0
            }
    
    async def search_industry_trends(self, job_title: str, 
                                   locations: List[str]) -> Dict[str, Any]:
        """Search for salary trends across multiple locations"""
        try:
            # Search for each location concurrently
            tasks = []
            for location in locations[:5]:  # Limit to 5 locations
                tasks.append(self.aggregate_salary_data(job_title, location))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            location_data = {}
            for i, result in enumerate(results):
                if isinstance(result, dict) and "aggregated_salary" in result:
                    location_data[locations[i]] = result["aggregated_salary"]
            
            # Calculate trends
            if location_data:
                all_averages = [data["average"] for data in location_data.values() 
                              if data and data.get("average")]
                
                return {
                    "job_title": job_title,
                    "locations_analyzed": list(location_data.keys()),
                    "salary_by_location": location_data,
                    "trends": {
                        "highest_paying_location": max(location_data.items(), 
                                                     key=lambda x: x[1].get("average", 0))[0],
                        "lowest_paying_location": min(location_data.items(), 
                                                    key=lambda x: x[1].get("average", float('inf')))[0],
                        "average_across_locations": sum(all_averages) // len(all_averages) if all_averages else None,
                        "salary_variance": max(all_averages) - min(all_averages) if all_averages else None
                    }
                }
            
            return {
                "error": "No trend data available",
                "locations_analyzed": []
            }
            
        except Exception as e:
            logger.error(f"Error searching industry trends: {e}")
            return {"error": str(e)}
    
    async def get_market_competitiveness(self, current_salary: int, job_title: str, 
                                       location: str, experience_years: int = None) -> Dict[str, Any]:
        """Analyze how competitive a salary is in the market"""
        try:
            # Get market data
            market_data = await self.aggregate_salary_data(
                job_title, location, experience_years
            )
            
            if "aggregated_salary" not in market_data:
                return {"error": "Unable to determine market competitiveness"}
            
            agg_salary = market_data["aggregated_salary"]
            
            # Calculate percentile
            percentile = None
            if all([agg_salary.get("min"), agg_salary.get("max")]):
                salary_range = agg_salary["max"] - agg_salary["min"]
                if salary_range > 0:
                    percentile = ((current_salary - agg_salary["min"]) / salary_range) * 100
                    percentile = max(0, min(100, percentile))  # Clamp to 0-100
            
            # Determine competitiveness
            if percentile is not None:
                if percentile >= 75:
                    competitiveness = "Highly Competitive"
                elif percentile >= 50:
                    competitiveness = "Competitive"
                elif percentile >= 25:
                    competitiveness = "Below Market"
                else:
                    competitiveness = "Significantly Below Market"
            else:
                competitiveness = "Unable to determine"
            
            return {
                "current_salary": current_salary,
                "market_data": market_data,
                "percentile": percentile,
                "competitiveness": competitiveness,
                "recommendations": {
                    "target_salary": agg_salary.get("average"),
                    "negotiation_range": {
                        "min": int(agg_salary.get("average", 0) * 0.9) if agg_salary.get("average") else None,
                        "max": int(agg_salary.get("average", 0) * 1.1) if agg_salary.get("average") else None
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market competitiveness: {e}")
            return {"error": str(e)}


# Singleton instance
_async_scraper = None


async def get_async_scraper() -> AsyncSalaryWebScraper:
    """Get or create async scraper singleton"""
    global _async_scraper
    if _async_scraper is None:
        _async_scraper = AsyncSalaryWebScraper()
    return _async_scraper
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List, Dict, Any, Optional
import time
import re
import logging
from config import settings

logger = logging.getLogger(__name__)

class SalaryWebScraper:
    """Web scraper for salary research"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.driver = None
    
    def _init_selenium_driver(self):
        """Initialize Selenium driver for dynamic content"""
        if not self.driver:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            
            try:
                self.driver = webdriver.Chrome(options=options)
            except Exception as e:
                logger.error(f"Failed to initialize Selenium driver: {e}")
                self.driver = None
    
    def search_glassdoor_salaries(self, job_title: str, location: str) -> Dict[str, Any]:
        """Search Glassdoor for salary information (mock implementation)"""
        try:
            # Note: In production, you would need proper API access or respect robots.txt
            # This is a simplified example structure
            search_url = f"https://www.glassdoor.com/Salaries/{location}-{job_title}-salary"
            
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
    
    def search_indeed_salaries(self, job_title: str, location: str) -> Dict[str, Any]:
        """Search Indeed for salary information"""
        try:
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
    
    def search_payscale_salaries(self, job_title: str, location: str, 
                                experience_years: int = None) -> Dict[str, Any]:
        """Search PayScale for salary information"""
        try:
            # Mock implementation
            base_salary = 90000
            if experience_years:
                # Adjust based on experience
                if experience_years < 3:
                    multiplier = 0.8
                elif experience_years < 7:
                    multiplier = 1.0
                elif experience_years < 12:
                    multiplier = 1.3
                else:
                    multiplier = 1.5
                    
                base_salary = int(base_salary * multiplier)
            
            return {
                "source": "PayScale",
                "title": job_title,
                "location": location,
                "experience_level": experience_years,
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
    
    def search_linkedin_salaries(self, job_title: str, location: str) -> Dict[str, Any]:
        """Search LinkedIn for salary insights"""
        try:
            # Mock implementation
            return {
                "source": "LinkedIn",
                "title": job_title,
                "location": location,
                "salary_range": {
                    "min": 85000,
                    "max": 125000,
                    "average": 105000
                },
                "top_paying_companies": [
                    {"name": "Tech Corp", "average": 130000},
                    {"name": "Finance Inc", "average": 125000},
                    {"name": "Startup XYZ", "average": 110000}
                ],
                "skills_impact": {
                    "Machine Learning": "+15%",
                    "Cloud Computing": "+12%",
                    "Leadership": "+20%"
                },
                "last_updated": "2024-01-14"
            }
            
        except Exception as e:
            logger.error(f"Error searching LinkedIn: {e}")
            return {}
    
    def aggregate_salary_data(self, job_title: str, location: str, 
                            experience_years: int = None) -> Dict[str, Any]:
        """Aggregate salary data from multiple sources"""
        sources_data = []
        
        # Collect data from multiple sources
        glassdoor_data = self.search_glassdoor_salaries(job_title, location)
        if glassdoor_data:
            sources_data.append(glassdoor_data)
        
        indeed_data = self.search_indeed_salaries(job_title, location)
        if indeed_data:
            sources_data.append(indeed_data)
        
        payscale_data = self.search_payscale_salaries(job_title, location, experience_years)
        if payscale_data:
            sources_data.append(payscale_data)
        
        linkedin_data = self.search_linkedin_salaries(job_title, location)
        if linkedin_data:
            sources_data.append(linkedin_data)
        
        if not sources_data:
            return {
                "error": "No salary data found",
                "sources_checked": 4
            }
        
        # Calculate aggregated values
        all_mins = [s["salary_range"]["min"] for s in sources_data if "salary_range" in s]
        all_maxs = [s["salary_range"]["max"] for s in sources_data if "salary_range" in s]
        all_avgs = [s["salary_range"]["average"] for s in sources_data if "salary_range" in s]
        
        return {
            "job_title": job_title,
            "location": location,
            "experience_years": experience_years,
            "aggregated_salary": {
                "min": int(sum(all_mins) / len(all_mins)) if all_mins else 0,
                "max": int(sum(all_maxs) / len(all_maxs)) if all_maxs else 0,
                "average": int(sum(all_avgs) / len(all_avgs)) if all_avgs else 0,
                "median": int(sorted(all_avgs)[len(all_avgs)//2]) if all_avgs else 0
            },
            "sources": sources_data,
            "confidence_score": len(sources_data) / 4,  # Based on number of sources
            "market_insights": self._generate_market_insights(sources_data)
        }
    
    def _generate_market_insights(self, sources_data: List[Dict]) -> List[str]:
        """Generate insights from salary data"""
        insights = []
        
        # Salary variance insight
        all_avgs = [s["salary_range"]["average"] for s in sources_data if "salary_range" in s]
        if all_avgs:
            variance = max(all_avgs) - min(all_avgs)
            if variance > 20000:
                insights.append(f"High salary variance across sources (${variance:,} difference)")
            else:
                insights.append("Consistent salary estimates across sources")
        
        # Top paying companies insight
        for source in sources_data:
            if "top_paying_companies" in source:
                top_company = source["top_paying_companies"][0]
                insights.append(f"Top paying company: {top_company['name']} (${top_company['average']:,})")
                break
        
        # Skills impact insight
        for source in sources_data:
            if "skills_impact" in source:
                top_skill = max(source["skills_impact"].items(), key=lambda x: x[1])
                insights.append(f"Most valuable skill: {top_skill[0]} ({top_skill[1]} salary increase)")
                break
        
        return insights
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            self.driver = None
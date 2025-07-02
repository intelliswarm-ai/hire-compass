from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.tools import Tool
from agents.base_agent import BaseAgent
from models.schemas import SalaryResearch, ExperienceLevel
from tools.web_scraper import SalaryWebScraper
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SalaryResearchAgent(BaseAgent):
    """Agent responsible for researching market salaries"""
    
    def __init__(self):
        super().__init__("SalaryResearch")
        self.web_scraper = SalaryWebScraper()
        self.tools = self.create_tools()
        self.prompt = self.create_prompt()
        self.agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    def create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a salary research specialist with expertise in compensation analysis.
            
            Your task is to:
            1. Research current market salaries for specific positions and locations
            2. Consider factors like experience level, skills, and industry
            3. Provide salary ranges with confidence levels
            4. Compare candidate expectations with market rates
            5. Suggest negotiation strategies
            
            Use the web scraping tools to gather data from multiple sources.
            Provide data-driven insights and recommendations.
            """),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
    
    def create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="search_salaries",
                func=self._search_salaries,
                description="Search for salary data across multiple sources"
            ),
            Tool(
                name="analyze_compensation",
                func=self._analyze_compensation,
                description="Analyze compensation data and provide insights"
            ),
            Tool(
                name="calculate_total_compensation",
                func=self._calculate_total_compensation,
                description="Calculate total compensation including benefits"
            ),
            Tool(
                name="generate_negotiation_strategy",
                func=self._generate_negotiation_strategy,
                description="Generate salary negotiation recommendations"
            )
        ]
    
    def _search_salaries(self, job_title: str, location: str, 
                        experience_years: int = None) -> Dict[str, Any]:
        """Search for salary data using web scraper"""
        try:
            return self.web_scraper.aggregate_salary_data(
                job_title=job_title,
                location=location,
                experience_years=experience_years
            )
        except Exception as e:
            logger.error(f"Error searching salaries: {e}")
            return {"error": str(e)}
    
    def _analyze_compensation(self, salary_data: Dict[str, Any], 
                            candidate_expectation: float = None) -> Dict[str, Any]:
        """Analyze compensation data and candidate fit"""
        analysis = {
            "market_position": "",
            "competitiveness": "",
            "recommendation": ""
        }
        
        if "aggregated_salary" not in salary_data:
            return {"error": "No salary data available"}
        
        market_avg = salary_data["aggregated_salary"]["average"]
        market_min = salary_data["aggregated_salary"]["min"]
        market_max = salary_data["aggregated_salary"]["max"]
        
        if candidate_expectation:
            if candidate_expectation < market_min:
                analysis["market_position"] = "below_market"
                analysis["competitiveness"] = "highly_competitive"
                analysis["recommendation"] = f"Candidate expectations are below market. Consider offering ${market_min:,} - ${market_avg:,}"
            elif candidate_expectation > market_max:
                analysis["market_position"] = "above_market"
                analysis["competitiveness"] = "challenging"
                analysis["recommendation"] = f"Candidate expectations exceed market rates. Market range is ${market_avg:,} - ${market_max:,}"
            else:
                analysis["market_position"] = "within_market"
                analysis["competitiveness"] = "competitive"
                percentile = ((candidate_expectation - market_min) / (market_max - market_min)) * 100
                analysis["recommendation"] = f"Expectations align with market ({percentile:.0f}th percentile)"
        
        analysis["market_summary"] = {
            "average": market_avg,
            "range": f"${market_min:,} - ${market_max:,}",
            "data_sources": len(salary_data.get("sources", []))
        }
        
        return analysis
    
    def _calculate_total_compensation(self, base_salary: float, 
                                    benefits: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate total compensation package value"""
        total_comp = base_salary
        benefits_value = 0
        
        if benefits:
            # Standard benefits calculations
            if benefits.get("health_insurance"):
                benefits_value += 6000  # Average employer contribution
            
            if benefits.get("401k_match"):
                match_percent = benefits.get("401k_match_percent", 3)
                benefits_value += base_salary * (match_percent / 100)
            
            if benefits.get("bonus_target"):
                bonus_percent = benefits.get("bonus_target_percent", 10)
                benefits_value += base_salary * (bonus_percent / 100)
            
            if benefits.get("equity"):
                benefits_value += benefits.get("equity_value", 0)
            
            if benefits.get("pto_days"):
                daily_rate = base_salary / 260  # Working days per year
                benefits_value += daily_rate * benefits.get("pto_days", 15)
        
        total_comp += benefits_value
        
        return {
            "base_salary": base_salary,
            "benefits_value": benefits_value,
            "total_compensation": total_comp,
            "benefits_percentage": (benefits_value / base_salary) * 100 if base_salary > 0 else 0
        }
    
    def _generate_negotiation_strategy(self, candidate_data: Dict[str, Any],
                                     position_data: Dict[str, Any],
                                     market_data: Dict[str, Any]) -> List[str]:
        """Generate salary negotiation recommendations"""
        strategies = []
        
        # Experience-based negotiation
        candidate_years = candidate_data.get("total_experience_years", 0)
        position_min_years = position_data.get("min_experience_years", 0)
        
        if candidate_years > position_min_years + 3:
            strategies.append(
                "Leverage extensive experience to negotiate higher end of salary range"
            )
        
        # Skills-based negotiation
        candidate_skills = set([s["name"].lower() for s in candidate_data.get("skills", [])])
        preferred_skills = set([s.lower() for s in position_data.get("preferred_skills", [])])
        
        if len(candidate_skills.intersection(preferred_skills)) > len(preferred_skills) * 0.7:
            strategies.append(
                "Highlight rare skill combinations to justify premium compensation"
            )
        
        # Market-based negotiation
        if market_data.get("aggregated_salary"):
            market_avg = market_data["aggregated_salary"]["average"]
            if position_data.get("salary_range_max", 0) < market_avg:
                strategies.append(
                    f"Present market data showing average of ${market_avg:,} to support higher offer"
                )
        
        # Location-based negotiation
        if position_data.get("work_mode") == "remote":
            strategies.append(
                "If relocating, negotiate for relocation assistance or cost-of-living adjustment"
            )
        
        # Alternative compensation
        strategies.append(
            "Consider negotiating for signing bonus, equity, or additional PTO if base salary is fixed"
        )
        
        return strategies
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process salary research request"""
        try:
            position_title = input_data.get('position_title')
            location = input_data.get('location')
            experience_years = input_data.get('experience_years', 5)
            candidate_expectation = input_data.get('candidate_expectation')
            
            if not position_title or not location:
                raise ValueError("position_title and location are required")
            
            # Search for salary data
            salary_data = self._search_salaries(
                job_title=position_title,
                location=location,
                experience_years=experience_years
            )
            
            if "error" in salary_data:
                raise Exception(salary_data["error"])
            
            # Analyze compensation
            analysis = self._analyze_compensation(
                salary_data=salary_data,
                candidate_expectation=candidate_expectation
            )
            
            # Generate negotiation strategies if candidate data provided
            strategies = []
            if input_data.get('candidate_data') and input_data.get('position_data'):
                strategies = self._generate_negotiation_strategy(
                    candidate_data=input_data['candidate_data'],
                    position_data=input_data['position_data'],
                    market_data=salary_data
                )
            
            # Create SalaryResearch object
            experience_level = self._determine_experience_level(experience_years)
            
            salary_research = SalaryResearch(
                position_title=position_title,
                location=location,
                experience_level=experience_level,
                market_average=salary_data["aggregated_salary"]["average"],
                market_min=salary_data["aggregated_salary"]["min"],
                market_max=salary_data["aggregated_salary"]["max"],
                sources=salary_data.get("sources", [])
            )
            
            result = {
                "salary_research": salary_research.dict(),
                "analysis": analysis,
                "market_insights": salary_data.get("market_insights", []),
                "negotiation_strategies": strategies,
                "confidence_score": salary_data.get("confidence_score", 0)
            }
            
            logger.info(f"Completed salary research for {position_title} in {location}")
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Error in salary research: {str(e)}")
            return {"success": False, "error": str(e)}
        finally:
            # Cleanup resources
            self.web_scraper.cleanup()
    
    def _determine_experience_level(self, years: int) -> ExperienceLevel:
        """Determine experience level based on years"""
        if years < 2:
            return ExperienceLevel.ENTRY
        elif years < 5:
            return ExperienceLevel.MID
        elif years < 8:
            return ExperienceLevel.SENIOR
        elif years < 12:
            return ExperienceLevel.LEAD
        else:
            return ExperienceLevel.EXECUTIVE
    
    def batch_research(self, positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform batch salary research for multiple positions"""
        results = []
        
        for position in positions:
            try:
                result = self.process({
                    "position_title": position.get("title"),
                    "location": position.get("location"),
                    "experience_years": position.get("min_experience_years", 5)
                })
                
                if result["success"]:
                    results.append({
                        "position_id": position.get("id"),
                        "salary_data": result["result"]
                    })
            except Exception as e:
                logger.error(f"Error researching salary for position {position.get('id')}: {e}")
                continue
        
        return results
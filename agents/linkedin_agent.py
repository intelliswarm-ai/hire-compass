"""
LinkedIn Integration Agent.

This agent specializes in fetching and analyzing job positions from LinkedIn,
matching them with resumes, and providing comprehensive job market insights.
"""

import json
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory

from agents.base_agent import BaseAgent
from tools.linkedin_integration import create_linkedin_tools
from src.shared.protocols import Cache, Logger


class LinkedInAgent(BaseAgent):
    """Agent specialized in LinkedIn job analysis and matching."""
    
    def __init__(
        self,
        name: str = "LinkedIn Agent",
        model_name: Optional[str] = None,
        mcp_server_url: str = "http://localhost:8002",
        advanced_analyzer_url: str = "http://localhost:8001",
        cache: Optional[Cache] = None,
        logger: Optional[Logger] = None
    ):
        super().__init__(name, model_name)
        self.mcp_server_url = mcp_server_url
        self.advanced_analyzer_url = advanced_analyzer_url
        self.cache = cache
        self.logger = logger
        
        # Create LinkedIn-specific tools
        self.tools = create_linkedin_tools(
            mcp_server_url=mcp_server_url,
            advanced_analyzer_url=advanced_analyzer_url,
            cache=cache,
            logger=logger
        )
        
        # Initialize agent with tools
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the agent with LinkedIn tools."""
        # Create memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self._create_agent_prompt(),
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def _create_agent_prompt(self):
        """Create specialized prompt for LinkedIn agent."""
        from langchain.agents import create_react_agent
        from langchain.prompts import PromptTemplate
        
        template = """You are a LinkedIn Job Market Specialist with expertise in:
        - Fetching job positions from specific companies on LinkedIn
        - Analyzing job requirements and trends
        - Matching resumes with available positions
        - Providing strategic career advice based on market data

        You have access to the following tools:
        {tools}

        When analyzing jobs or matching resumes:
        1. Always fetch fresh job data from LinkedIn when requested
        2. Provide detailed analysis of job requirements
        3. Calculate match scores based on skills, experience, and preferences
        4. Offer actionable recommendations for improving matches
        5. Identify skill gaps and suggest learning paths

        Current conversation:
        {chat_history}

        User request: {input}

        Use the tools to gather information and provide comprehensive insights.
        Always explain your findings clearly and provide specific recommendations.

        {agent_scratchpad}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["input", "tools", "agent_scratchpad", "chat_history"]
        )
        
        return create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
    
    async def analyze_company_opportunities(
        self,
        company_name: str,
        resume_text: Optional[str] = None,
        location: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze job opportunities at a specific company.
        
        Args:
            company_name: Name of the company
            resume_text: Optional resume for matching
            location: Preferred location
            preferences: Job preferences (type, level, etc.)
        
        Returns:
            Comprehensive analysis of opportunities
        """
        # Build the analysis request
        request_parts = [f"Analyze job opportunities at {company_name}"]
        
        if location:
            request_parts.append(f"in {location}")
        
        if preferences:
            if preferences.get("job_type"):
                request_parts.append(f"for {preferences['job_type']} positions")
            if preferences.get("experience_level"):
                request_parts.append(f"at {preferences['experience_level']} level")
        
        request = " ".join(request_parts)
        
        if resume_text:
            request += f"\n\nAlso match these positions with my resume:\n{resume_text[:500]}..."
        
        # Get analysis from agent
        response = self.agent_executor.run(input=request)
        
        # Parse and structure the response
        return self._parse_analysis_response(response, company_name)
    
    async def compare_companies(
        self,
        resume_text: str,
        company_list: List[str],
        criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compare job opportunities across multiple companies.
        
        Args:
            resume_text: Resume for matching
            company_list: List of companies to compare
            criteria: Comparison criteria
        
        Returns:
            Comparative analysis across companies
        """
        request = f"""Compare job opportunities for me across these companies: {', '.join(company_list)}
        
        My resume:
        {resume_text[:500]}...
        
        Please analyze:
        1. Number and quality of matching positions at each company
        2. Best overall matches across all companies
        3. Company-specific requirements and culture fit
        4. Recommendations on which companies to prioritize
        """
        
        if criteria:
            request += f"\n\nFocus on these criteria: {json.dumps(criteria)}"
        
        response = self.agent_executor.run(input=request)
        
        return self._parse_comparison_response(response, company_list)
    
    async def get_market_insights(
        self,
        role: str,
        companies: Optional[List[str]] = None,
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get market insights for a specific role.
        
        Args:
            role: Job role/title
            companies: Optional list of companies to analyze
            location: Optional location filter
        
        Returns:
            Market insights and trends
        """
        request = f"Provide market insights for {role} positions"
        
        if companies:
            request += f" at companies: {', '.join(companies)}"
        
        if location:
            request += f" in {location}"
        
        request += """
        
        Include:
        1. Common requirements and skills
        2. Experience levels in demand
        3. Salary ranges (if available)
        4. Growth trends
        5. Recommendations for job seekers
        """
        
        response = self.agent_executor.run(input=request)
        
        return self._parse_insights_response(response, role)
    
    async def optimize_job_search(
        self,
        resume_text: str,
        target_companies: List[str],
        target_role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize job search strategy.
        
        Args:
            resume_text: Current resume
            target_companies: List of target companies
            target_role: Optional specific role
        
        Returns:
            Optimized job search strategy
        """
        request = f"""Help me optimize my job search strategy.
        
        Target companies: {', '.join(target_companies)}
        {'Target role: ' + target_role if target_role else ''}
        
        My resume:
        {resume_text[:500]}...
        
        Please provide:
        1. Best matching positions at each company
        2. Skills I should highlight or develop
        3. Companies where I have the strongest matches
        4. Application prioritization strategy
        5. Networking recommendations
        """
        
        response = self.agent_executor.run(input=request)
        
        return self._parse_strategy_response(response)
    
    def _parse_analysis_response(
        self,
        response: str,
        company_name: str
    ) -> Dict[str, Any]:
        """Parse analysis response into structured format."""
        # This is a simplified parser - in production, use NLP
        return {
            "company": company_name,
            "analysis": response,
            "summary": {
                "total_positions": self._extract_number(response, "positions"),
                "match_rate": self._extract_percentage(response, "match"),
                "top_skills": self._extract_skills(response),
                "recommendations": self._extract_recommendations(response)
            }
        }
    
    def _parse_comparison_response(
        self,
        response: str,
        companies: List[str]
    ) -> Dict[str, Any]:
        """Parse comparison response."""
        return {
            "companies_analyzed": companies,
            "comparison": response,
            "rankings": self._extract_rankings(response, companies),
            "best_matches": self._extract_best_matches(response),
            "recommendations": self._extract_recommendations(response)
        }
    
    def _parse_insights_response(
        self,
        response: str,
        role: str
    ) -> Dict[str, Any]:
        """Parse market insights response."""
        return {
            "role": role,
            "insights": response,
            "key_findings": {
                "required_skills": self._extract_skills(response),
                "experience_levels": self._extract_experience_levels(response),
                "trends": self._extract_trends(response)
            }
        }
    
    def _parse_strategy_response(self, response: str) -> Dict[str, Any]:
        """Parse job search strategy response."""
        return {
            "strategy": response,
            "action_items": self._extract_action_items(response),
            "priority_companies": self._extract_priority_items(response, "companies"),
            "priority_skills": self._extract_priority_items(response, "skills")
        }
    
    # Utility methods for parsing
    def _extract_number(self, text: str, keyword: str) -> Optional[int]:
        """Extract number associated with keyword."""
        import re
        pattern = rf"(\d+)\s*{keyword}"
        match = re.search(pattern, text, re.IGNORECASE)
        return int(match.group(1)) if match else None
    
    def _extract_percentage(self, text: str, keyword: str) -> Optional[float]:
        """Extract percentage associated with keyword."""
        import re
        pattern = rf"(\d+\.?\d*)\s*%?\s*{keyword}"
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else None
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills mentioned in text."""
        # Common skill keywords
        skill_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'aws',
            'docker', 'kubernetes', 'sql', 'machine learning', 'data science'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in skill_keywords:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from text."""
        recommendations = []
        
        # Look for recommendation patterns
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in ['recommend', 'suggest', 'should', 'consider']):
                if len(line.strip()) > 20:  # Meaningful recommendation
                    recommendations.append(line.strip())
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _extract_rankings(self, text: str, companies: List[str]) -> Dict[str, int]:
        """Extract company rankings from text."""
        rankings = {}
        
        for i, company in enumerate(companies):
            # Look for ranking mentions
            if company.lower() in text.lower():
                # Simple heuristic - order of appearance
                rankings[company] = i + 1
        
        return rankings
    
    def _extract_best_matches(self, text: str) -> List[str]:
        """Extract best job matches from text."""
        matches = []
        
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['match', 'score', 'fit']):
                if '%' in line or any(word in line.lower() for word in ['excellent', 'good', 'strong']):
                    matches.append(line.strip())
        
        return matches[:10]
    
    def _extract_experience_levels(self, text: str) -> List[str]:
        """Extract experience levels from text."""
        levels = []
        
        level_keywords = [
            'entry level', 'junior', 'mid-level', 'senior', 'lead', 'principal',
            'staff', 'director', 'executive'
        ]
        
        text_lower = text.lower()
        for level in level_keywords:
            if level in text_lower:
                levels.append(level)
        
        return levels
    
    def _extract_trends(self, text: str) -> List[str]:
        """Extract market trends from text."""
        trends = []
        
        trend_keywords = ['growing', 'increasing', 'demand', 'trend', 'popular', 'emerging']
        
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in trend_keywords):
                if len(line.strip()) > 20:
                    trends.append(line.strip())
        
        return trends[:5]
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from text."""
        action_items = []
        
        # Look for action-oriented language
        action_keywords = ['apply', 'update', 'learn', 'network', 'prepare', 'focus']
        
        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in action_keywords):
                if len(line.strip()) > 15:
                    action_items.append(line.strip())
        
        return action_items
    
    def _extract_priority_items(self, text: str, item_type: str) -> List[str]:
        """Extract priority items of specified type."""
        items = []
        
        # Look for priority mentions
        lines = text.split('\n')
        for line in lines:
            if item_type.lower() in line.lower() and any(
                keyword in line.lower() for keyword in ['priority', 'focus', 'top', 'best']
            ):
                items.append(line.strip())
        
        return items[:5]
"""
Async resume parser agent with improved performance.
"""

from typing import Dict, Any, List
import asyncio
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.output_parsers import JsonOutputParser
from agents.base_agent import BaseAgent
from models.schemas import Resume, Education, Experience, Skill
from tools.async_document_loaders import AsyncResumeLoader
import json
import re
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncResumeParserAgent(BaseAgent):
    """Async agent for parsing resumes with high performance"""
    
    def __init__(self):
        super().__init__("AsyncResumeParser")
        self.output_parser = JsonOutputParser()
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
        self.resume_loader = AsyncResumeLoader()
        self._executor_pool = ThreadPoolExecutor(max_workers=2)
    
    def create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume parser. Extract structured information from resumes.
            
            Your task is to analyze the resume text and extract:
            1. Personal information (name, email, phone, location)
            2. Professional summary
            3. Education details (degree, field, institution, year)
            4. Work experience (company, position, duration, description, technologies)
            5. Skills with proficiency levels
            6. Certifications
            7. Languages spoken
            8. Total years of experience
            9. Current and expected salary (if mentioned)
            
            Format the output as a structured JSON object following the Resume schema.
            Be accurate and don't make up information that's not in the resume.
            """),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
    
    def create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="extract_email",
                func=self._extract_email,
                description="Extract email address from text",
                coroutine=self._extract_email_async
            ),
            Tool(
                name="extract_phone",
                func=self._extract_phone,
                description="Extract phone number from text",
                coroutine=self._extract_phone_async
            ),
            Tool(
                name="calculate_experience",
                func=self._calculate_total_experience,
                description="Calculate total years of experience from work history",
                coroutine=self._calculate_total_experience_async
            ),
            Tool(
                name="parse_skills",
                func=self._parse_skills,
                description="Parse and categorize skills from resume text",
                coroutine=self._parse_skills_async
            )
        ]
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process resume parsing request asynchronously"""
        try:
            file_path = input_data.get("file_path")
            resume_id = input_data.get("resume_id", str(datetime.now().timestamp()))
            
            if not file_path:
                return {
                    "success": False,
                    "error": "No file path provided"
                }
            
            # Load resume asynchronously
            resume_data = await self.resume_loader.load_resume(file_path)
            
            # Extract text
            resume_text = resume_data.get("raw_text", "")
            
            if not resume_text:
                return {
                    "success": False,
                    "error": "Could not extract text from resume"
                }
            
            # Run extraction tasks concurrently
            extraction_tasks = [
                self._extract_personal_info_async(resume_text),
                self._extract_education_async(resume_text),
                self._extract_experience_async(resume_text),
                self._extract_skills_async(resume_text)
            ]
            
            results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
            
            # Combine results
            personal_info = results[0] if not isinstance(results[0], Exception) else {}
            education = results[1] if not isinstance(results[1], Exception) else []
            experience = results[2] if not isinstance(results[2], Exception) else []
            skills = results[3] if not isinstance(results[3], Exception) else []
            
            # Calculate total experience
            total_experience = await self._calculate_total_experience_async(experience)
            
            # Create structured resume
            resume = {
                "id": resume_id,
                "name": personal_info.get("name", "Unknown"),
                "email": personal_info.get("email", ""),
                "phone": personal_info.get("phone", ""),
                "location": personal_info.get("location", ""),
                "summary": personal_info.get("summary", ""),
                "total_experience_years": total_experience,
                "education": education,
                "experience": experience,
                "skills": skills,
                "languages": personal_info.get("languages", []),
                "certifications": personal_info.get("certifications", [])
            }
            
            return {
                "success": True,
                "resume": resume,
                "raw_text": resume_text
            }
            
        except Exception as e:
            logger.error(f"Error parsing resume: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _extract_email_async(self, text: str) -> str:
        """Extract email from text asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_email, text)
    
    def _extract_email(self, text: str) -> str:
        """Extract email from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else ""
    
    async def _extract_phone_async(self, text: str) -> str:
        """Extract phone number asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_phone, text)
    
    def _extract_phone(self, text: str) -> str:
        """Extract phone number from text"""
        phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}'
        matches = re.findall(phone_pattern, text)
        return matches[0] if matches else ""
    
    async def _extract_personal_info_async(self, text: str) -> Dict[str, Any]:
        """Extract personal information asynchronously"""
        # Run multiple extractions concurrently
        tasks = [
            self._extract_email_async(text),
            self._extract_phone_async(text),
            self._extract_name_async(text),
            self._extract_location_async(text),
            self._extract_summary_async(text)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            "email": results[0],
            "phone": results[1],
            "name": results[2],
            "location": results[3],
            "summary": results[4]
        }
    
    async def _extract_name_async(self, text: str) -> str:
        """Extract name from resume text"""
        # Simple heuristic: first line often contains name
        lines = text.strip().split('\n')
        if lines:
            # Filter out common non-name patterns
            first_line = lines[0].strip()
            if not any(word in first_line.lower() for word in ['resume', 'cv', 'curriculum']):
                return first_line
        return "Unknown"
    
    async def _extract_location_async(self, text: str) -> str:
        """Extract location from resume"""
        # Look for city, state patterns
        location_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})'
        matches = re.findall(location_pattern, text)
        if matches:
            return f"{matches[0][0]}, {matches[0][1]}"
        return ""
    
    async def _extract_summary_async(self, text: str) -> str:
        """Extract professional summary"""
        # Look for summary section
        summary_patterns = [
            r'(?i)(?:professional\s+)?summary[:\s]*([^\n]+(?:\n(?![A-Z]{2,})[^\n]+)*)',
            r'(?i)objective[:\s]*([^\n]+(?:\n(?![A-Z]{2,})[^\n]+)*)',
            r'(?i)profile[:\s]*([^\n]+(?:\n(?![A-Z]{2,})[^\n]+)*)'
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ""
    
    async def _extract_education_async(self, text: str) -> List[Dict[str, Any]]:
        """Extract education information"""
        education = []
        
        # Look for education section
        education_section = re.search(
            r'(?i)education[:\s]*\n((?:(?![A-Z]{2,}:)[^\n]+\n?)+)',
            text
        )
        
        if education_section:
            edu_text = education_section.group(1)
            # Parse individual education entries
            # This is simplified - would need more sophisticated parsing
            lines = edu_text.strip().split('\n')
            
            current_edu = {}
            for line in lines:
                if 'bachelor' in line.lower() or 'master' in line.lower() or 'phd' in line.lower():
                    if current_edu:
                        education.append(current_edu)
                    current_edu = {"degree": line.strip()}
                elif current_edu and not current_edu.get("institution"):
                    current_edu["institution"] = line.strip()
            
            if current_edu:
                education.append(current_edu)
        
        return education
    
    async def _extract_experience_async(self, text: str) -> List[Dict[str, Any]]:
        """Extract work experience"""
        experience = []
        
        # Look for experience section
        exp_section = re.search(
            r'(?i)(?:work\s+)?experience[:\s]*\n((?:(?![A-Z]{2,}:)[^\n]+\n?)+)',
            text,
            re.MULTILINE
        )
        
        if exp_section:
            exp_text = exp_section.group(1)
            # Parse experience entries (simplified)
            entries = re.split(r'\n(?=[A-Z][a-z]+.*\||\d{4})', exp_text)
            
            for entry in entries:
                if entry.strip():
                    exp_dict = {
                        "description": entry.strip(),
                        "duration_months": 12  # Placeholder
                    }
                    experience.append(exp_dict)
        
        return experience
    
    async def _extract_skills_async(self, text: str) -> List[Dict[str, Any]]:
        """Extract skills from resume"""
        return await self._parse_skills_async(text)
    
    async def _parse_skills_async(self, text: str) -> List[Dict[str, Any]]:
        """Parse and categorize skills asynchronously"""
        skills = []
        
        # Common skill keywords
        skill_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'go', 'rust', 'typescript'],
            'frameworks': ['django', 'flask', 'react', 'angular', 'vue', 'spring', 'fastapi'],
            'databases': ['sql', 'postgresql', 'mongodb', 'redis', 'elasticsearch'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'tools': ['git', 'jenkins', 'jira', 'terraform', 'ansible']
        }
        
        text_lower = text.lower()
        
        for category, keywords in skill_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    skills.append({
                        "name": keyword.title(),
                        "category": category,
                        "level": "Intermediate"  # Would need better logic
                    })
        
        return skills
    
    def _parse_skills(self, text: str) -> List[Dict[str, Any]]:
        """Synchronous version of parse skills"""
        return asyncio.run(self._parse_skills_async(text))
    
    async def _calculate_total_experience_async(self, experiences: List[Dict]) -> float:
        """Calculate total years of experience asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._calculate_total_experience,
            experiences
        )
    
    def _calculate_total_experience(self, experiences: List[Dict]) -> float:
        """Calculate total years of experience"""
        total_months = 0
        for exp in experiences:
            if 'duration_months' in exp:
                total_months += exp['duration_months']
        return round(total_months / 12, 1)
    
    async def batch_parse_resumes(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Parse multiple resumes concurrently"""
        tasks = []
        for file_path in file_paths:
            task = self.process({"file_path": file_path})
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        parsed_resumes = []
        for i, result in enumerate(results):
            if isinstance(result, dict) and result.get("success"):
                parsed_resumes.append(result)
            else:
                logger.error(f"Failed to parse resume {file_paths[i]}: {result}")
        
        return parsed_resumes
    
    def __del__(self):
        """Cleanup executor pool"""
        if hasattr(self, '_executor_pool'):
            self._executor_pool.shutdown(wait=False)
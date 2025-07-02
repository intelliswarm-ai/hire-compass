from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.tools import Tool
from agents.base_agent import BaseAgent
from models.schemas import JobPosition, ExperienceLevel, WorkMode, EducationLevel
from tools.document_loaders import ResumeLoader
from tools.vector_store import VectorStoreManager
import json
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class JobParserAgent(BaseAgent):
    """Agent responsible for parsing job descriptions and storing them in vector database"""
    
    def __init__(self):
        super().__init__("JobParser")
        self.vector_store = VectorStoreManager()
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
            ("system", """You are an expert job description parser. Extract structured information from job postings.
            
            Your task is to analyze the job description and extract:
            1. Job title
            2. Department
            3. Location and work mode (onsite/remote/hybrid)
            4. Detailed description
            5. Key responsibilities (as a list)
            6. Requirements (must-have qualifications)
            7. Preferred qualifications (nice-to-have)
            8. Required skills
            9. Preferred skills
            10. Experience level (entry/mid/senior/lead/executive)
            11. Minimum and maximum years of experience required
            12. Education requirements
            13. Salary range (if mentioned)
            
            Format the output as structured JSON following the JobPosition schema.
            Be accurate and extract only what's explicitly mentioned.
            
            For experience level, use these guidelines:
            - Entry: 0-2 years, fresh graduates, junior positions
            - Mid: 3-5 years, individual contributor roles
            - Senior: 5-8 years, senior individual contributor
            - Lead: 8-12 years, team lead, architect roles
            - Executive: 12+ years, director, VP, C-level
            """),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
    
    def create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="parse_experience_level",
                func=self._parse_experience_level,
                description="Parse experience level from job description"
            ),
            Tool(
                name="extract_skills",
                func=self._extract_skills,
                description="Extract skills from text, separating required and preferred"
            ),
            Tool(
                name="parse_salary_range",
                func=self._parse_salary_range,
                description="Extract salary range from text"
            ),
            Tool(
                name="parse_work_mode",
                func=self._parse_work_mode,
                description="Determine work mode (onsite/remote/hybrid)"
            )
        ]
    
    def _parse_experience_level(self, text: str) -> str:
        """Determine experience level from job description"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['senior', 'sr.', 'principal', 'staff']):
            return ExperienceLevel.SENIOR
        elif any(word in text_lower for word in ['lead', 'manager', 'architect']):
            return ExperienceLevel.LEAD
        elif any(word in text_lower for word in ['director', 'vp', 'vice president', 'executive']):
            return ExperienceLevel.EXECUTIVE
        elif any(word in text_lower for word in ['junior', 'entry', 'graduate', 'fresher']):
            return ExperienceLevel.ENTRY
        else:
            return ExperienceLevel.MID
    
    def _extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract required and preferred skills"""
        skills = {"required": [], "preferred": []}
        
        # Common skill patterns
        skill_patterns = [
            r"(?:required|must have).*?skills?:?\s*(.*?)(?:\n\n|\Z)",
            r"(?:preferred|nice to have).*?skills?:?\s*(.*?)(?:\n\n|\Z)",
            r"technologies?:?\s*(.*?)(?:\n\n|\Z)"
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Extract individual skills
                skills_text = match.strip()
                individual_skills = re.split(r'[,;•\n]', skills_text)
                
                for skill in individual_skills:
                    skill = skill.strip(' -•*')
                    if skill and len(skill) > 2:
                        if 'preferred' in pattern.lower() or 'nice' in pattern.lower():
                            skills["preferred"].append(skill)
                        else:
                            skills["required"].append(skill)
        
        return skills
    
    def _parse_salary_range(self, text: str) -> Dict[str, float]:
        """Extract salary range from text"""
        salary_patterns = [
            r'\$(\d+)[kK]?\s*-\s*\$(\d+)[kK]?',
            r'(\d+)[kK]\s*-\s*(\d+)[kK]',
            r'(\d+),(\d+)\s*-\s*(\d+),(\d+)'
        ]
        
        for pattern in salary_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    min_sal = float(groups[0]) * (1000 if 'k' in text.lower() else 1)
                    max_sal = float(groups[1]) * (1000 if 'k' in text.lower() else 1)
                    return {"min": min_sal, "max": max_sal}
        
        return {"min": None, "max": None}
    
    def _parse_work_mode(self, text: str) -> str:
        """Determine work mode from job description"""
        text_lower = text.lower()
        
        if 'remote' in text_lower and 'hybrid' not in text_lower:
            return WorkMode.REMOTE
        elif 'hybrid' in text_lower:
            return WorkMode.HYBRID
        else:
            return WorkMode.ONSITE
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process job description and store in vector database"""
        try:
            job_text = input_data.get('job_description', '')
            file_path = input_data.get('file_path')
            
            # Load from file if provided
            if file_path:
                documents = ResumeLoader.load_document(file_path)
                job_text = "\n".join([doc.page_content for doc in documents])
            
            if not job_text:
                raise ValueError("job_description or file_path is required")
            
            # Use LLM to parse job description
            result = self.executor.invoke({
                "input": f"Parse this job description and extract all relevant information:\n\n{job_text}"
            })
            
            # Parse the output
            parsed_data = self._parse_llm_output(result['output'])
            
            # Extract additional information using tools
            skills = self._extract_skills(job_text)
            salary = self._parse_salary_range(job_text)
            
            # Create JobPosition object
            position = JobPosition(
                id=input_data.get('position_id', f"pos_{datetime.now().timestamp()}"),
                title=parsed_data.get('title', 'Unknown Position'),
                department=parsed_data.get('department', 'General'),
                location=parsed_data.get('location', 'Not Specified'),
                work_mode=self._parse_work_mode(job_text),
                description=parsed_data.get('description', job_text[:500]),
                responsibilities=parsed_data.get('responsibilities', []),
                requirements=parsed_data.get('requirements', []),
                preferred_qualifications=parsed_data.get('preferred_qualifications', []),
                required_skills=skills['required'] or parsed_data.get('required_skills', []),
                preferred_skills=skills['preferred'] or parsed_data.get('preferred_skills', []),
                experience_level=self._parse_experience_level(job_text),
                min_experience_years=parsed_data.get('min_experience_years', 0),
                max_experience_years=parsed_data.get('max_experience_years'),
                education_requirements=self._parse_education_requirements(parsed_data.get('education_requirements', [])),
                salary_range_min=salary['min'] or parsed_data.get('salary_range_min'),
                salary_range_max=salary['max'] or parsed_data.get('salary_range_max'),
                is_active=True
            )
            
            # Store in vector database
            self.vector_store.add_position(position.dict())
            
            logger.info(f"Successfully parsed and stored job position: {position.title}")
            return {"success": True, "position": position.dict()}
            
        except Exception as e:
            logger.error(f"Error parsing job description: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _parse_llm_output(self, output: str) -> Dict:
        """Parse LLM output to extract JSON data"""
        try:
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return self._fallback_parse(output)
        except json.JSONDecodeError:
            return self._fallback_parse(output)
    
    def _fallback_parse(self, text: str) -> Dict:
        """Fallback parsing when JSON extraction fails"""
        lines = text.split('\n')
        title = lines[0] if lines else "Unknown Position"
        
        return {
            "title": title,
            "description": text[:500],
            "requirements": [],
            "responsibilities": []
        }
    
    def _parse_education_requirements(self, education_data: List[str]) -> List[EducationLevel]:
        """Parse education requirements"""
        levels = []
        for edu in education_data:
            edu_lower = edu.lower()
            if 'high school' in edu_lower:
                levels.append(EducationLevel.HIGH_SCHOOL)
            elif 'bachelor' in edu_lower or 'bs' in edu_lower or 'ba' in edu_lower:
                levels.append(EducationLevel.BACHELORS)
            elif 'master' in edu_lower or 'ms' in edu_lower or 'ma' in edu_lower:
                levels.append(EducationLevel.MASTERS)
            elif 'phd' in edu_lower or 'doctorate' in edu_lower:
                levels.append(EducationLevel.PHD)
        
        return levels if levels else [EducationLevel.BACHELORS]
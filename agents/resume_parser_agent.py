from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.output_parsers import JsonOutputParser
from agents.base_agent import BaseAgent
from models.schemas import Resume, Education, Experience, Skill
from tools.document_loaders import ResumeLoader
import json
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ResumeParserAgent(BaseAgent):
    """Agent responsible for parsing resumes and extracting structured information"""
    
    def __init__(self):
        super().__init__("ResumeParser")
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
                description="Extract email address from text"
            ),
            Tool(
                name="extract_phone",
                func=self._extract_phone,
                description="Extract phone number from text"
            ),
            Tool(
                name="calculate_experience",
                func=self._calculate_total_experience,
                description="Calculate total years of experience from work history"
            ),
            Tool(
                name="parse_date",
                func=self._parse_date,
                description="Parse date strings into datetime objects"
            )
        ]
    
    def _extract_email(self, text: str) -> str:
        """Extract email from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else ""
    
    def _extract_phone(self, text: str) -> str:
        """Extract phone number from text"""
        phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}'
        matches = re.findall(phone_pattern, text)
        return matches[0] if matches else ""
    
    def _calculate_total_experience(self, experiences: List[Dict]) -> float:
        """Calculate total years of experience"""
        total_months = 0
        for exp in experiences:
            if 'duration_months' in exp:
                total_months += exp['duration_months']
        return round(total_months / 12, 1)
    
    def _parse_date(self, date_str: str) -> str:
        """Parse various date formats"""
        date_formats = [
            "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y",
            "%B %Y", "%b %Y", "%Y"
        ]
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).isoformat()
            except ValueError:
                continue
        return ""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process resume file and extract structured information"""
        try:
            file_path = input_data.get('file_path')
            if not file_path:
                raise ValueError("file_path is required")
            
            # Load document
            documents = ResumeLoader.load_document(file_path)
            raw_text = "\n".join([doc.page_content for doc in documents])
            
            # Use LLM to parse resume
            result = self.executor.invoke({
                "input": f"Parse this resume and extract all relevant information:\n\n{raw_text}"
            })
            
            # Parse the output
            parsed_data = self._parse_llm_output(result['output'])
            
            # Create Resume object
            resume = Resume(
                id=input_data.get('resume_id', f"resume_{datetime.now().timestamp()}"),
                name=parsed_data.get('name', 'Unknown'),
                email=parsed_data.get('email', ''),
                phone=parsed_data.get('phone'),
                location=parsed_data.get('location'),
                summary=parsed_data.get('summary'),
                education=self._parse_education(parsed_data.get('education', [])),
                experience=self._parse_experience(parsed_data.get('experience', [])),
                skills=self._parse_skills(parsed_data.get('skills', [])),
                certifications=parsed_data.get('certifications', []),
                languages=parsed_data.get('languages', []),
                total_experience_years=parsed_data.get('total_experience_years', 0),
                current_salary=parsed_data.get('current_salary'),
                expected_salary=parsed_data.get('expected_salary'),
                raw_text=raw_text
            )
            
            logger.info(f"Successfully parsed resume for {resume.name}")
            return {"success": True, "resume": resume.dict()}
            
        except Exception as e:
            logger.error(f"Error parsing resume: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _parse_llm_output(self, output: str) -> Dict:
        """Parse LLM output to extract JSON data"""
        try:
            # Try to extract JSON from the output
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing logic
                return self._fallback_parse(output)
        except json.JSONDecodeError:
            return self._fallback_parse(output)
    
    def _fallback_parse(self, text: str) -> Dict:
        """Fallback parsing when JSON extraction fails"""
        return {
            "name": "Unknown",
            "email": self._extract_email(text),
            "phone": self._extract_phone(text),
            "raw_text": text
        }
    
    def _parse_education(self, education_data: List[Dict]) -> List[Education]:
        """Parse education data into Education objects"""
        education_list = []
        for edu in education_data:
            try:
                education_list.append(Education(**edu))
            except Exception as e:
                logger.warning(f"Failed to parse education entry: {e}")
        return education_list
    
    def _parse_experience(self, experience_data: List[Dict]) -> List[Experience]:
        """Parse experience data into Experience objects"""
        experience_list = []
        for exp in experience_data:
            try:
                experience_list.append(Experience(**exp))
            except Exception as e:
                logger.warning(f"Failed to parse experience entry: {e}")
        return experience_list
    
    def _parse_skills(self, skills_data: List[Any]) -> List[Skill]:
        """Parse skills data into Skill objects"""
        skills_list = []
        for skill in skills_data:
            try:
                if isinstance(skill, str):
                    skills_list.append(Skill(name=skill))
                elif isinstance(skill, dict):
                    skills_list.append(Skill(**skill))
            except Exception as e:
                logger.warning(f"Failed to parse skill entry: {e}")
        return skills_list
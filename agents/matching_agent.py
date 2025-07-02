from typing import Dict, Any, List, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.tools import Tool
from agents.base_agent import BaseAgent
from models.schemas import MatchResult, Resume, JobPosition
from tools.vector_store import VectorStoreManager
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MatchingAgent(BaseAgent):
    """Agent responsible for matching resumes with job positions"""
    
    def __init__(self):
        super().__init__("Matching")
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
            ("system", """You are an expert HR matching specialist. Analyze resumes and job positions to find the best matches.
            
            Your task is to:
            1. Compare skills, experience, and qualifications
            2. Calculate match scores for different aspects
            3. Identify strengths and gaps
            4. Provide actionable recommendations
            
            Consider these factors:
            - Skill alignment (both required and preferred)
            - Experience level and years
            - Education requirements
            - Location compatibility
            - Salary expectations
            - Career aspirations
            
            Be objective and provide detailed analysis with specific examples.
            """),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
    
    def create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="calculate_skill_match",
                func=self._calculate_skill_match,
                description="Calculate skill match score between resume and position"
            ),
            Tool(
                name="calculate_experience_match",
                func=self._calculate_experience_match,
                description="Calculate experience match score"
            ),
            Tool(
                name="calculate_education_match",
                func=self._calculate_education_match,
                description="Calculate education match score"
            ),
            Tool(
                name="analyze_gaps",
                func=self._analyze_gaps,
                description="Identify gaps between resume and position requirements"
            )
        ]
    
    def _calculate_skill_match(self, resume_skills: List[str], required_skills: List[str], 
                              preferred_skills: List[str]) -> Dict[str, Any]:
        """Calculate skill match score"""
        resume_skills_lower = [s.lower() for s in resume_skills]
        required_skills_lower = [s.lower() for s in required_skills]
        preferred_skills_lower = [s.lower() for s in preferred_skills]
        
        # Check required skills
        required_matches = sum(1 for skill in required_skills_lower if skill in resume_skills_lower)
        required_score = required_matches / len(required_skills_lower) if required_skills_lower else 1.0
        
        # Check preferred skills
        preferred_matches = sum(1 for skill in preferred_skills_lower if skill in resume_skills_lower)
        preferred_score = preferred_matches / len(preferred_skills_lower) if preferred_skills_lower else 0.0
        
        # Overall skill score (70% required, 30% preferred)
        overall_score = (required_score * 0.7) + (preferred_score * 0.3)
        
        return {
            "overall_score": overall_score,
            "required_match_rate": required_score,
            "preferred_match_rate": preferred_score,
            "matched_required": [s for s in required_skills if s.lower() in resume_skills_lower],
            "missing_required": [s for s in required_skills if s.lower() not in resume_skills_lower],
            "matched_preferred": [s for s in preferred_skills if s.lower() in resume_skills_lower]
        }
    
    def _calculate_experience_match(self, resume_years: float, min_years: float, 
                                   max_years: float = None) -> Dict[str, Any]:
        """Calculate experience match score"""
        if resume_years < min_years:
            # Under-qualified
            score = max(0, 1 - (min_years - resume_years) / min_years)
            status = "under-qualified"
        elif max_years and resume_years > max_years:
            # Over-qualified
            score = max(0.7, 1 - (resume_years - max_years) / (max_years * 2))
            status = "over-qualified"
        else:
            # Within range
            score = 1.0
            status = "perfect-match"
        
        return {
            "score": score,
            "status": status,
            "resume_years": resume_years,
            "required_range": f"{min_years}-{max_years or 'unlimited'} years"
        }
    
    def _calculate_education_match(self, resume_education: List[Dict], 
                                 required_education: List[str]) -> Dict[str, Any]:
        """Calculate education match score"""
        education_hierarchy = {
            "high_school": 1,
            "bachelors": 2,
            "masters": 3,
            "phd": 4
        }
        
        # Get highest education level from resume
        resume_levels = [edu.get("level", "other") for edu in resume_education]
        resume_max_level = max([education_hierarchy.get(level, 0) for level in resume_levels], default=0)
        
        # Get required education level
        required_max_level = max([education_hierarchy.get(level, 0) for level in required_education], default=0)
        
        if resume_max_level >= required_max_level:
            score = 1.0
            status = "meets-requirements"
        else:
            score = resume_max_level / required_max_level if required_max_level > 0 else 0.5
            status = "below-requirements"
        
        return {
            "score": score,
            "status": status,
            "resume_education": resume_levels,
            "required_education": required_education
        }
    
    def _analyze_gaps(self, resume_data: Dict, position_data: Dict) -> List[str]:
        """Identify gaps between resume and position"""
        gaps = []
        
        # Skill gaps
        resume_skills = [s["name"].lower() for s in resume_data.get("skills", [])]
        required_skills = [s.lower() for s in position_data.get("required_skills", [])]
        missing_skills = [s for s in required_skills if s not in resume_skills]
        
        if missing_skills:
            gaps.append(f"Missing required skills: {', '.join(missing_skills[:5])}")
        
        # Experience gap
        if resume_data["total_experience_years"] < position_data["min_experience_years"]:
            gap_years = position_data["min_experience_years"] - resume_data["total_experience_years"]
            gaps.append(f"Needs {gap_years:.1f} more years of experience")
        
        # Education gap
        resume_education = [edu.get("level") for edu in resume_data.get("education", [])]
        required_education = position_data.get("education_requirements", [])
        if required_education and not any(edu in resume_education for edu in required_education):
            gaps.append(f"Education requirement not met: {', '.join(required_education)}")
        
        # Location gap
        if resume_data.get("location") and position_data.get("location"):
            if resume_data["location"].lower() != position_data["location"].lower():
                if position_data.get("work_mode") == "onsite":
                    gaps.append(f"Location mismatch: candidate in {resume_data['location']}, position in {position_data['location']}")
        
        return gaps
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process matching request for resume and position"""
        try:
            resume_data = input_data.get('resume')
            position_data = input_data.get('position')
            
            if not resume_data or not position_data:
                raise ValueError("Both resume and position data are required")
            
            # Calculate individual match scores
            skill_match = self._calculate_skill_match(
                [s["name"] for s in resume_data.get("skills", [])],
                position_data.get("required_skills", []),
                position_data.get("preferred_skills", [])
            )
            
            experience_match = self._calculate_experience_match(
                resume_data["total_experience_years"],
                position_data["min_experience_years"],
                position_data.get("max_experience_years")
            )
            
            education_match = self._calculate_education_match(
                resume_data.get("education", []),
                position_data.get("education_requirements", [])
            )
            
            # Calculate salary compatibility
            salary_score = self._calculate_salary_compatibility(
                resume_data.get("expected_salary"),
                position_data.get("salary_range_min"),
                position_data.get("salary_range_max")
            )
            
            # Identify gaps and strengths
            gaps = self._analyze_gaps(resume_data, position_data)
            strengths = self._identify_strengths(resume_data, position_data, skill_match)
            
            # Calculate overall score (weighted average)
            overall_score = (
                skill_match["overall_score"] * 0.4 +
                experience_match["score"] * 0.3 +
                education_match["score"] * 0.2 +
                salary_score * 0.1
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                resume_data, position_data, skill_match, experience_match, gaps
            )
            
            # Create match result
            match_result = MatchResult(
                resume_id=resume_data["id"],
                position_id=position_data["id"],
                overall_score=overall_score,
                skill_match_score=skill_match["overall_score"],
                experience_match_score=experience_match["score"],
                education_match_score=education_match["score"],
                aspiration_match_score=0.0,  # Will be implemented with aspiration agent
                salary_compatibility_score=salary_score,
                strengths=strengths,
                gaps=gaps,
                recommendations=recommendations,
                detailed_analysis={
                    "skill_analysis": skill_match,
                    "experience_analysis": experience_match,
                    "education_analysis": education_match
                }
            )
            
            logger.info(f"Completed matching: {resume_data['name']} for {position_data['title']} - Score: {overall_score:.2f}")
            return {"success": True, "match_result": match_result.dict()}
            
        except Exception as e:
            logger.error(f"Error in matching process: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _calculate_salary_compatibility(self, expected_salary: float = None, 
                                      min_salary: float = None, max_salary: float = None) -> float:
        """Calculate salary compatibility score"""
        if not expected_salary or not min_salary:
            return 0.5  # Neutral score if data not available
        
        if expected_salary < min_salary:
            # Candidate expects less than minimum
            return 1.0
        elif max_salary and expected_salary > max_salary:
            # Candidate expects more than maximum
            return max(0, 1 - (expected_salary - max_salary) / max_salary)
        else:
            # Within range
            return 1.0
    
    def _identify_strengths(self, resume_data: Dict, position_data: Dict, 
                          skill_match: Dict) -> List[str]:
        """Identify candidate strengths for the position"""
        strengths = []
        
        # Strong skill match
        if skill_match["overall_score"] > 0.8:
            strengths.append(f"Excellent skill match ({skill_match['overall_score']*100:.0f}%)")
        
        # Matching experience level
        if (resume_data["total_experience_years"] >= position_data["min_experience_years"] and 
            (not position_data.get("max_experience_years") or 
             resume_data["total_experience_years"] <= position_data["max_experience_years"])):
            strengths.append(f"Experience level perfectly matches requirements")
        
        # Additional skills
        candidate_skills = [s["name"].lower() for s in resume_data.get("skills", [])]
        extra_relevant_skills = [s for s in candidate_skills 
                               if s in [ps.lower() for ps in position_data.get("preferred_skills", [])]]
        if extra_relevant_skills:
            strengths.append(f"Has preferred skills: {', '.join(extra_relevant_skills[:3])}")
        
        # Location match
        if (resume_data.get("location", "").lower() == position_data.get("location", "").lower()):
            strengths.append("Located in the same area as the position")
        
        return strengths
    
    def _generate_recommendations(self, resume_data: Dict, position_data: Dict,
                                skill_match: Dict, experience_match: Dict, 
                                gaps: List[str]) -> List[str]:
        """Generate recommendations for the match"""
        recommendations = []
        
        # Skill recommendations
        if skill_match["missing_required"]:
            recommendations.append(
                f"Candidate should highlight transferable skills or similar technologies to: "
                f"{', '.join(skill_match['missing_required'][:3])}"
            )
        
        # Experience recommendations
        if experience_match["status"] == "under-qualified":
            recommendations.append(
                "Consider candidate's growth potential and learning ability"
            )
        elif experience_match["status"] == "over-qualified":
            recommendations.append(
                "Discuss career goals to ensure role aligns with candidate's expectations"
            )
        
        # Interview focus areas
        if skill_match["overall_score"] > 0.7:
            recommendations.append(
                "Focus interview on cultural fit and soft skills"
            )
        else:
            recommendations.append(
                "Technical assessment recommended to verify skill proficiency"
            )
        
        # Salary discussion
        if resume_data.get("expected_salary") and position_data.get("salary_range_max"):
            if resume_data["expected_salary"] > position_data["salary_range_max"] * 0.9:
                recommendations.append(
                    "Prepare to discuss compensation expectations and total benefits package"
                )
        
        return recommendations
    
    def batch_match(self, resumes: List[Dict], positions: List[Dict]) -> List[MatchResult]:
        """Perform batch matching of multiple resumes against multiple positions"""
        results = []
        
        for resume in resumes:
            for position in positions:
                match_result = self.process({
                    "resume": resume,
                    "position": position
                })
                
                if match_result["success"]:
                    results.append(match_result["match_result"])
        
        # Sort by overall score
        results.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return results
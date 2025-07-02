from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.tools import Tool
from agents.base_agent import BaseAgent
from agents.resume_parser_agent import ResumeParserAgent
from agents.job_parser_agent import JobParserAgent
from agents.matching_agent import MatchingAgent
from agents.salary_research_agent import SalaryResearchAgent
from agents.aspiration_agent import AspirationAgent
from models.schemas import BatchMatchRequest, BatchMatchResponse, MatchResult
from tools.vector_store import VectorStoreManager
import time
import logging
from config import settings

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """Master agent that coordinates all other agents for the matching process"""
    
    def __init__(self):
        super().__init__("Orchestrator")
        
        # Initialize all sub-agents
        self.resume_parser = ResumeParserAgent()
        self.job_parser = JobParserAgent()
        self.matching_agent = MatchingAgent()
        self.salary_agent = SalaryResearchAgent()
        self.aspiration_agent = AspirationAgent()
        
        # Vector store for efficient retrieval
        self.vector_store = VectorStoreManager()
        
        # Thread pool for parallel processing
        self.executor_pool = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_agents
        )
        
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
            max_iterations=5
        )
    
    def create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are the master orchestrator for an HR matching system.
            
            Your responsibilities:
            1. Coordinate multiple specialized agents for resume-job matching
            2. Optimize the matching process for scale (up to 300 positions)
            3. Ensure efficient use of resources and parallel processing
            4. Aggregate results and provide comprehensive matching reports
            5. Handle errors gracefully and ensure system reliability
            
            Process flow:
            1. Parse resumes and job descriptions
            2. Store in vector database for efficient retrieval
            3. Perform initial similarity matching
            4. Deep match analysis for top candidates
            5. Research salaries for matched positions
            6. Analyze career aspirations
            7. Generate final recommendations
            
            Always prioritize accuracy and provide actionable insights for HR teams.
            """),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
    
    def create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="parse_resume",
                func=self._parse_resume,
                description="Parse a resume file"
            ),
            Tool(
                name="parse_job",
                func=self._parse_job,
                description="Parse a job description"
            ),
            Tool(
                name="match_candidates",
                func=self._match_candidates,
                description="Match candidates to positions"
            ),
            Tool(
                name="research_salaries",
                func=self._research_salaries,
                description="Research market salaries"
            ),
            Tool(
                name="analyze_aspirations",
                func=self._analyze_aspirations,
                description="Analyze candidate aspirations"
            )
        ]
    
    def _parse_resume(self, file_path: str, resume_id: Optional[str] = None) -> Dict[str, Any]:
        """Parse a resume using the resume parser agent"""
        return self.resume_parser.process({
            "file_path": file_path,
            "resume_id": resume_id
        })
    
    def _parse_job(self, job_description: str = None, file_path: str = None, 
                   position_id: Optional[str] = None) -> Dict[str, Any]:
        """Parse a job description using the job parser agent"""
        return self.job_parser.process({
            "job_description": job_description,
            "file_path": file_path,
            "position_id": position_id
        })
    
    def _match_candidates(self, resume_data: Dict[str, Any], 
                         position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Match a candidate to a position"""
        return self.matching_agent.process({
            "resume": resume_data,
            "position": position_data
        })
    
    def _research_salaries(self, position_title: str, location: str,
                          experience_years: int = None) -> Dict[str, Any]:
        """Research salaries for a position"""
        return self.salary_agent.process({
            "position_title": position_title,
            "location": location,
            "experience_years": experience_years
        })
    
    def _analyze_aspirations(self, resume_data: Dict[str, Any],
                           position_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze candidate career aspirations"""
        return self.aspiration_agent.process({
            "resume_data": resume_data,
            "position_data": position_data
        })
    
    def process_single_match(self, resume_path: str, position_path: str,
                           include_salary: bool = True,
                           include_aspirations: bool = True) -> Dict[str, Any]:
        """Process a single resume-position match"""
        try:
            # Parse resume
            resume_result = self._parse_resume(resume_path)
            if not resume_result["success"]:
                return {"success": False, "error": f"Failed to parse resume: {resume_result['error']}"}
            
            resume_data = resume_result["resume"]
            
            # Parse position
            position_result = self._parse_job(file_path=position_path)
            if not position_result["success"]:
                return {"success": False, "error": f"Failed to parse position: {position_result['error']}"}
            
            position_data = position_result["position"]
            
            # Perform matching
            match_result = self._match_candidates(resume_data, position_data)
            if not match_result["success"]:
                return {"success": False, "error": f"Matching failed: {match_result['error']}"}
            
            result = {
                "success": True,
                "match": match_result["match_result"],
                "resume": resume_data,
                "position": position_data
            }
            
            # Optional: Research salaries
            if include_salary:
                salary_result = self._research_salaries(
                    position_title=position_data["title"],
                    location=position_data["location"],
                    experience_years=int(resume_data["total_experience_years"])
                )
                if salary_result["success"]:
                    result["salary_research"] = salary_result["result"]
            
            # Optional: Analyze aspirations
            if include_aspirations:
                aspiration_result = self._analyze_aspirations(resume_data, position_data)
                if aspiration_result["success"]:
                    result["aspiration_analysis"] = aspiration_result["result"]
                    # Update match score with aspiration alignment
                    if "position_alignment" in aspiration_result["result"]:
                        result["match"]["aspiration_match_score"] = \
                            aspiration_result["result"]["position_alignment"]["overall_score"]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single match processing: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def process_batch_match(self, batch_request: BatchMatchRequest) -> BatchMatchResponse:
        """Process batch matching request for multiple resumes and positions"""
        start_time = time.time()
        all_results = []
        
        try:
            # Retrieve resume and position data from vector store
            resumes = self._retrieve_resumes(batch_request.resume_ids)
            positions = self._retrieve_positions(batch_request.position_ids)
            
            # First pass: Use vector similarity for initial filtering
            initial_matches = self._vector_similarity_match(resumes, positions)
            
            # Second pass: Deep analysis for top matches
            futures = []
            
            with self.executor_pool as executor:
                for match in initial_matches[:100]:  # Limit deep analysis to top 100
                    future = executor.submit(
                        self._deep_match_analysis,
                        match["resume"],
                        match["position"],
                        batch_request.include_salary_research,
                        batch_request.include_aspiration_analysis
                    )
                    futures.append(future)
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")
            
            # Sort by overall score
            all_results.sort(key=lambda x: x.overall_score, reverse=True)
            
            processing_time = time.time() - start_time
            
            return BatchMatchResponse(
                results=all_results,
                processing_time_seconds=processing_time,
                total_comparisons=len(resumes) * len(positions)
            )
            
        except Exception as e:
            logger.error(f"Error in batch match processing: {str(e)}")
            raise
    
    def _retrieve_resumes(self, resume_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve resume data from storage"""
        # In production, this would retrieve from database
        # For now, return empty list
        return []
    
    def _retrieve_positions(self, position_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve position data from storage"""
        # In production, this would retrieve from database
        # For now, return empty list
        return []
    
    def _vector_similarity_match(self, resumes: List[Dict[str, Any]], 
                               positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform initial matching using vector similarity"""
        matches = []
        
        for resume in resumes:
            # Find similar positions for this resume
            resume_text = self._create_resume_search_text(resume)
            similar_positions = self.vector_store.search_similar_positions(
                resume_text=resume_text,
                k=10  # Top 10 positions per resume
            )
            
            for pos_match in similar_positions:
                # Find the full position data
                position = next((p for p in positions if p["id"] == pos_match["position_id"]), None)
                if position:
                    matches.append({
                        "resume": resume,
                        "position": position,
                        "similarity_score": pos_match["similarity_score"]
                    })
        
        # Sort by similarity score
        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return matches
    
    def _deep_match_analysis(self, resume: Dict[str, Any], position: Dict[str, Any],
                           include_salary: bool, include_aspirations: bool) -> MatchResult:
        """Perform deep analysis for a resume-position pair"""
        try:
            # Detailed matching
            match_result = self._match_candidates(resume, position)
            if not match_result["success"]:
                return None
            
            match_data = match_result["match_result"]
            
            # Enhance with salary research
            if include_salary:
                salary_result = self._research_salaries(
                    position_title=position["title"],
                    location=position["location"],
                    experience_years=int(resume["total_experience_years"])
                )
                if salary_result["success"]:
                    salary_data = salary_result["result"]
                    # Update salary compatibility score
                    if salary_data["analysis"]["competitiveness"] == "highly_competitive":
                        match_data["salary_compatibility_score"] = 1.0
                    elif salary_data["analysis"]["competitiveness"] == "competitive":
                        match_data["salary_compatibility_score"] = 0.8
                    else:
                        match_data["salary_compatibility_score"] = 0.5
            
            # Enhance with aspiration analysis
            if include_aspirations:
                aspiration_result = self._analyze_aspirations(resume, position)
                if aspiration_result["success"]:
                    aspiration_data = aspiration_result["result"]
                    if "position_alignment" in aspiration_data:
                        match_data["aspiration_match_score"] = \
                            aspiration_data["position_alignment"]["overall_score"]
            
            # Recalculate overall score with all components
            match_data["overall_score"] = self._calculate_final_score(match_data)
            
            return MatchResult(**match_data)
            
        except Exception as e:
            logger.error(f"Error in deep match analysis: {e}")
            return None
    
    def _create_resume_search_text(self, resume: Dict[str, Any]) -> str:
        """Create searchable text from resume data"""
        parts = [
            resume.get("summary", ""),
            " ".join([s["name"] for s in resume.get("skills", [])]),
            " ".join([f"{e['position']} at {e['company']}" for e in resume.get("experience", [])])
        ]
        return " ".join(parts)
    
    def _calculate_final_score(self, match_data: Dict[str, Any]) -> float:
        """Calculate final overall score with all components"""
        weights = {
            "skill_match_score": 0.35,
            "experience_match_score": 0.25,
            "education_match_score": 0.15,
            "aspiration_match_score": 0.15,
            "salary_compatibility_score": 0.10
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for component, weight in weights.items():
            if component in match_data and match_data[component] is not None:
                total_score += match_data[component] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method for the orchestrator"""
        try:
            if "batch_request" in input_data:
                # Batch processing
                batch_request = BatchMatchRequest(**input_data["batch_request"])
                response = self.process_batch_match(batch_request)
                return {"success": True, "batch_response": response.dict()}
            else:
                # Single match processing
                return self.process_single_match(
                    resume_path=input_data["resume_path"],
                    position_path=input_data["position_path"],
                    include_salary=input_data.get("include_salary", True),
                    include_aspirations=input_data.get("include_aspirations", True)
                )
                
        except Exception as e:
            logger.error(f"Error in orchestrator processing: {str(e)}")
            return {"success": False, "error": str(e)}
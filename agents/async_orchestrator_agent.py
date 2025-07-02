"""
Async orchestrator agent for coordinating all other agents with high performance.
"""

import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

from agents.base_agent import BaseAgent
from agents.async_resume_parser_agent import AsyncResumeParserAgent
from tools.async_vector_store import get_async_vector_store
from tools.async_web_scraper import get_async_scraper
from models.schemas import MatchResult, BatchMatchRequest, BatchMatchResponse

logger = logging.getLogger(__name__)


class AsyncOrchestratorAgent(BaseAgent):
    """Async orchestrator that coordinates all agents for optimal performance"""
    
    def __init__(self):
        super().__init__("AsyncOrchestrator", model_name="llama2")
        self.resume_parser = AsyncResumeParserAgent()
        self.vector_store = None
        self.scraper = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize async components
        asyncio.create_task(self._init_async_components())
    
    async def _init_async_components(self):
        """Initialize async components"""
        self.vector_store = await get_async_vector_store()
        self.scraper = await get_async_scraper()
        logger.info("Async components initialized in orchestrator")
    
    def create_prompt(self):
        """Create orchestrator prompt"""
        return """You are an expert HR orchestrator coordinating multiple specialized agents.
        Your role is to:
        1. Understand the user's request
        2. Delegate tasks to appropriate agents
        3. Combine and synthesize results
        4. Provide comprehensive responses
        """
    
    def create_tools(self) -> list:
        """Orchestrator doesn't need direct tools"""
        return []
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process orchestration request asynchronously"""
        try:
            # Ensure async components are initialized
            if not self.vector_store or not self.scraper:
                await self._init_async_components()
            
            # Handle different request types
            if "batch_request" in input_data:
                return await self._process_batch_match(input_data["batch_request"])
            elif "single_match" in input_data:
                return await self._process_single_match(input_data["single_match"])
            elif "resume_analysis" in input_data:
                return await self._process_resume_analysis(input_data["resume_analysis"])
            elif "market_analysis" in input_data:
                return await self._process_market_analysis(input_data["market_analysis"])
            else:
                return await self._process_general_request(input_data)
                
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _process_batch_match(self, batch_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process batch matching request with concurrent execution"""
        try:
            start_time = datetime.now()
            
            resume_ids = batch_request.get("resume_ids", [])
            position_ids = batch_request.get("position_ids", [])
            top_k = batch_request.get("top_k", 10)
            
            # Create all matching tasks
            tasks = []
            for resume_id in resume_ids:
                for position_id in position_ids:
                    task = self._match_pair(resume_id, position_id)
                    tasks.append(task)
            
            # Execute all matches concurrently
            matches = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful matches
            valid_matches = []
            for match in matches:
                if isinstance(match, dict) and match.get("score", 0) > 0:
                    valid_matches.append(match)
            
            # Sort by score
            valid_matches.sort(key=lambda x: x["score"], reverse=True)
            
            # Apply top_k limit
            if top_k:
                valid_matches = valid_matches[:top_k]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "batch_response": {
                    "matches": valid_matches,
                    "total_comparisons": len(resume_ids) * len(position_ids),
                    "processing_time": processing_time,
                    "matches_found": len(valid_matches)
                }
            }
            
        except Exception as e:
            logger.error(f"Batch matching error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _match_pair(self, resume_id: str, position_id: str) -> Dict[str, Any]:
        """Match a single resume-position pair"""
        try:
            # This would fetch actual data from database
            # For now, using placeholder
            score = 0.75  # Would calculate actual score
            
            return {
                "resume_id": resume_id,
                "position_id": position_id,
                "score": score,
                "match_details": {
                    "skill_match": 0.8,
                    "experience_match": 0.7,
                    "education_match": 0.75,
                    "location_match": 0.9
                }
            }
        except Exception as e:
            logger.error(f"Error matching {resume_id} with {position_id}: {e}")
            return {"score": 0}
    
    async def _process_single_match(self, match_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process single match request"""
        try:
            resume_path = match_request.get("resume_path")
            position_path = match_request.get("position_path")
            
            # Parse documents concurrently
            tasks = [
                self.resume_parser.process({"file_path": resume_path}),
                self._parse_position_async(position_path)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            resume_result = results[0]
            position_result = results[1]
            
            if not isinstance(resume_result, dict) or not resume_result.get("success"):
                return {"success": False, "error": "Failed to parse resume"}
            
            if not isinstance(position_result, dict) or not position_result.get("success"):
                return {"success": False, "error": "Failed to parse position"}
            
            # Perform matching
            resume = resume_result["resume"]
            position = position_result["position"]
            
            # Vector similarity search
            resume_text = resume_result.get("raw_text", "")
            similar_positions = await self.vector_store.search_similar_positions(
                resume_text, k=1
            )
            
            # Calculate detailed match
            match_score = similar_positions[0]["similarity_score"] if similar_positions else 0.0
            
            # Add salary research if requested
            salary_data = None
            if match_request.get("include_salary", True):
                salary_data = await self.scraper.aggregate_salary_data(
                    job_title=position.get("title", ""),
                    location=position.get("location", ""),
                    experience_years=resume.get("total_experience_years", 0)
                )
            
            return {
                "success": True,
                "match": {
                    "overall_score": match_score,
                    "resume": resume,
                    "position": position,
                    "salary_research": salary_data
                }
            }
            
        except Exception as e:
            logger.error(f"Single match error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _parse_position_async(self, position_path: str) -> Dict[str, Any]:
        """Parse position document (placeholder)"""
        # This would use an async position parser
        return {
            "success": True,
            "position": {
                "title": "Software Engineer",
                "location": "San Francisco, CA",
                "requirements": ["Python", "FastAPI", "AWS"]
            }
        }
    
    async def _process_resume_analysis(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process comprehensive resume analysis"""
        try:
            resume_path = analysis_request.get("resume_path")
            
            # Parse resume
            resume_result = await self.resume_parser.process({"file_path": resume_path})
            
            if not resume_result.get("success"):
                return resume_result
            
            resume = resume_result["resume"]
            resume_text = resume_result["raw_text"]
            
            # Run multiple analyses concurrently
            tasks = [
                # Find matching positions
                self.vector_store.search_similar_positions(resume_text, k=10),
                
                # Market analysis for skills
                self._analyze_skill_demand(resume.get("skills", [])),
                
                # Salary research
                self.scraper.aggregate_salary_data(
                    job_title=self._infer_job_title(resume),
                    location=resume.get("location", ""),
                    experience_years=resume.get("total_experience_years", 0)
                ),
                
                # Career path suggestions
                self._suggest_career_paths(resume)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "success": True,
                "analysis": {
                    "resume": resume,
                    "matching_positions": results[0] if not isinstance(results[0], Exception) else [],
                    "skill_demand": results[1] if not isinstance(results[1], Exception) else {},
                    "salary_insights": results[2] if not isinstance(results[2], Exception) else {},
                    "career_paths": results[3] if not isinstance(results[3], Exception) else []
                }
            }
            
        except Exception as e:
            logger.error(f"Resume analysis error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_market_analysis(self, market_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process market analysis request"""
        try:
            job_title = market_request.get("job_title")
            locations = market_request.get("locations", [])
            
            # Analyze trends across locations
            trends = await self.scraper.search_industry_trends(job_title, locations)
            
            # Get competitive analysis
            tasks = []
            for location in locations[:5]:
                task = self.scraper.get_market_competitiveness(
                    current_salary=market_request.get("current_salary", 0),
                    job_title=job_title,
                    location=location,
                    experience_years=market_request.get("experience_years")
                )
                tasks.append(task)
            
            competitiveness_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "success": True,
                "market_analysis": {
                    "trends": trends,
                    "competitiveness": [
                        r for r in competitiveness_results 
                        if isinstance(r, dict) and "error" not in r
                    ]
                }
            }
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_skill_demand(self, skills: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze demand for specific skills"""
        # Placeholder implementation
        return {
            "high_demand": ["Python", "AWS", "Kubernetes"],
            "growing": ["Rust", "Go", "GraphQL"],
            "stable": ["Java", "SQL", "Git"]
        }
    
    async def _suggest_career_paths(self, resume: Dict[str, Any]) -> List[str]:
        """Suggest potential career paths"""
        # Placeholder implementation
        return [
            "Senior Software Engineer",
            "Technical Lead",
            "Solutions Architect",
            "Engineering Manager"
        ]
    
    def _infer_job_title(self, resume: Dict[str, Any]) -> str:
        """Infer current job title from resume"""
        # Simple heuristic - would be more sophisticated
        experience = resume.get("experience", [])
        if experience:
            return experience[0].get("position", "Software Engineer")
        return "Software Engineer"
    
    async def _process_general_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process general orchestration request"""
        # This would use LLM to understand and route the request
        return {
            "success": True,
            "message": "Request processed",
            "data": input_data
        }
    
    async def batch_process_resumes(self, resume_paths: List[str], 
                                  position_ids: List[str]) -> Dict[str, Any]:
        """Process multiple resumes against multiple positions"""
        try:
            # Parse all resumes concurrently
            resume_tasks = [
                self.resume_parser.process({"file_path": path}) 
                for path in resume_paths
            ]
            
            resume_results = await asyncio.gather(*resume_tasks, return_exceptions=True)
            
            # Store parsed resumes in vector store
            storage_tasks = []
            valid_resume_ids = []
            
            for result in resume_results:
                if isinstance(result, dict) and result.get("success"):
                    resume = result["resume"]
                    valid_resume_ids.append(resume["id"])
                    storage_tasks.append(self.vector_store.add_resume(resume))
            
            await asyncio.gather(*storage_tasks, return_exceptions=True)
            
            # Perform batch matching
            batch_request = {
                "resume_ids": valid_resume_ids,
                "position_ids": position_ids,
                "top_k": 50
            }
            
            return await self._process_batch_match(batch_request)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return {"success": False, "error": str(e)}
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
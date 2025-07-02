from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from mcp_server.models.resume_categorizer import Resume2PostCategorizer
from agents.resume_parser_agent import ResumeParserAgent
from agents.job_parser_agent import JobParserAgent

logger = logging.getLogger(__name__)

@dataclass
class CategorizationResult:
    """Result of resume categorization"""
    resume_id: str
    resume_name: str
    categorized_jobs: List[Dict[str, Any]]
    processing_time: float
    confidence_summary: Dict[str, int]

class Resume2PostTool:
    """
    MCP Tool for categorizing resumes to specific job posts
    Integrates with the existing multi-agent system
    """
    
    def __init__(self):
        self.categorizer = Resume2PostCategorizer()
        self.resume_parser = ResumeParserAgent()
        self.job_parser = JobParserAgent()
        
        # Try to load pre-trained model
        self.categorizer.load_model()
    
    async def categorize_resume_to_posts(
        self,
        resume_path: str,
        top_k: int = 10,
        min_confidence: str = "medium",
        filter_location: Optional[str] = None,
        filter_experience_level: Optional[str] = None
    ) -> CategorizationResult:
        """
        Categorize a resume to the most suitable job posts
        
        Args:
            resume_path: Path to the resume file
            top_k: Number of top job matches to return
            min_confidence: Minimum confidence level (low/medium/high)
            filter_location: Optional location filter
            filter_experience_level: Optional experience level filter
        
        Returns:
            CategorizationResult with matched jobs
        """
        import time
        start_time = time.time()
        
        try:
            # Parse resume
            logger.info(f"Parsing resume: {resume_path}")
            resume_result = self.resume_parser.process({
                "file_path": resume_path
            })
            
            if not resume_result["success"]:
                raise Exception(f"Failed to parse resume: {resume_result.get('error')}")
            
            resume_data = resume_result["resume"]
            
            # Categorize to job posts
            logger.info("Categorizing resume to job posts")
            categorized_jobs = self.categorizer.categorize_resume(
                resume_path=resume_path,
                top_k=top_k * 2  # Get more for filtering
            )
            
            # Apply filters
            filtered_jobs = self._apply_filters(
                categorized_jobs,
                min_confidence=min_confidence,
                filter_location=filter_location,
                filter_experience_level=filter_experience_level
            )
            
            # Limit to top_k
            filtered_jobs = filtered_jobs[:top_k]
            
            # Calculate confidence summary
            confidence_summary = {
                "high": sum(1 for job in filtered_jobs if job['confidence'] == 'high'),
                "medium": sum(1 for job in filtered_jobs if job['confidence'] == 'medium'),
                "low": sum(1 for job in filtered_jobs if job['confidence'] == 'low')
            }
            
            processing_time = time.time() - start_time
            
            return CategorizationResult(
                resume_id=resume_data["id"],
                resume_name=resume_data["name"],
                categorized_jobs=filtered_jobs,
                processing_time=processing_time,
                confidence_summary=confidence_summary
            )
            
        except Exception as e:
            logger.error(f"Error in categorization: {str(e)}")
            raise
    
    async def batch_categorize(
        self,
        resume_paths: List[str],
        top_k: int = 5,
        parallel: bool = True
    ) -> List[CategorizationResult]:
        """
        Categorize multiple resumes in batch
        
        Args:
            resume_paths: List of resume file paths
            top_k: Number of top matches per resume
            parallel: Process in parallel
        
        Returns:
            List of categorization results
        """
        results = []
        
        if parallel:
            import asyncio
            tasks = [
                self.categorize_resume_to_posts(path, top_k)
                for path in resume_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process {resume_paths[i]}: {result}")
                else:
                    valid_results.append(result)
            results = valid_results
        else:
            for path in resume_paths:
                try:
                    result = await self.categorize_resume_to_posts(path, top_k)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
        
        return results
    
    async def train_on_historical_data(
        self,
        match_history: List[Dict[str, Any]],
        min_score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Train the categorizer on historical match data
        
        Args:
            match_history: List of historical matches with scores
            min_score_threshold: Minimum score to consider as positive example
        
        Returns:
            Training summary
        """
        training_data = []
        
        for match in match_history:
            resume_data = match.get('resume_data')
            job_data = match.get('job_data')
            match_score = match.get('overall_score', 0)
            
            if resume_data and job_data:
                training_data.append((resume_data, job_data, match_score))
        
        if not training_data:
            return {"success": False, "error": "No valid training data"}
        
        # Train the model
        self.categorizer.train(training_data)
        
        return {
            "success": True,
            "training_samples": len(training_data),
            "positive_samples": sum(1 for _, _, score in training_data if score > min_score_threshold),
            "model_saved": True
        }
    
    async def update_model_weights(
        self,
        semantic_weight: float,
        feature_weight: float
    ) -> Dict[str, Any]:
        """
        Update the ensemble model weights
        
        Args:
            semantic_weight: Weight for semantic similarity (0-1)
            feature_weight: Weight for feature-based classification (0-1)
        
        Returns:
            Updated weights
        """
        self.categorizer.update_weights(semantic_weight, feature_weight)
        
        return {
            "semantic_weight": self.categorizer.semantic_weight,
            "feature_weight": self.categorizer.feature_weight
        }
    
    def _apply_filters(
        self,
        jobs: List[Dict[str, Any]],
        min_confidence: str,
        filter_location: Optional[str],
        filter_experience_level: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Apply filters to job matches"""
        filtered = jobs
        
        # Confidence filter
        confidence_levels = {"low": 0, "medium": 1, "high": 2}
        min_level = confidence_levels.get(min_confidence, 0)
        
        filtered = [
            job for job in filtered
            if confidence_levels.get(job['confidence'], 0) >= min_level
        ]
        
        # Location filter
        if filter_location:
            filtered = [
                job for job in filtered
                if filter_location.lower() in job.get('location', '').lower()
            ]
        
        # Experience level filter
        if filter_experience_level:
            # This would need job metadata about experience level
            # For now, keeping all jobs
            pass
        
        return filtered
    
    async def get_match_explanation(
        self,
        resume_path: str,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Get detailed explanation for why a resume matches a specific job
        
        Args:
            resume_path: Path to resume file
            job_id: ID of the job post
        
        Returns:
            Detailed match explanation
        """
        # Parse resume
        resume_result = self.resume_parser.process({"file_path": resume_path})
        if not resume_result["success"]:
            raise Exception("Failed to parse resume")
        
        resume_data = resume_result["resume"]
        
        # Get job data (would fetch from database in production)
        # For now, returning mock explanation
        
        return {
            "resume_id": resume_data["id"],
            "job_id": job_id,
            "match_factors": {
                "skills": {
                    "matched": ["Python", "Machine Learning", "AWS"],
                    "missing": ["Kubernetes", "Go"],
                    "score": 0.75
                },
                "experience": {
                    "required": "5+ years",
                    "candidate": f"{resume_data['total_experience_years']} years",
                    "score": 0.9
                },
                "education": {
                    "meets_requirement": True,
                    "score": 1.0
                },
                "location": {
                    "compatible": True,
                    "score": 1.0
                }
            },
            "overall_compatibility": 0.85,
            "recommendation": "Strong candidate - recommend interview"
        }
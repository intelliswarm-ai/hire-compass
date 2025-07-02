import os
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from mcp_server.tools.resume2post_tool import Resume2PostTool, CategorizationResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Resume2Post Categorizer")

# Initialize the Resume2Post tool
resume_tool = Resume2PostTool()

@mcp.tool(
    description="Categorize a resume to the most suitable job posts using AI"
)
async def categorize_resume(
    resume_path: str,
    top_k: int = 10,
    min_confidence: str = "medium",
    filter_location: Optional[str] = None,
    filter_experience_level: Optional[str] = None
) -> Dict[str, Any]:
    """
    Categorize a resume to find the most suitable job posts.
    
    Args:
        resume_path: Path to the resume file (PDF, DOCX, or TXT)
        top_k: Number of top job matches to return (default: 10)
        min_confidence: Minimum confidence level - 'low', 'medium', or 'high' (default: 'medium')
        filter_location: Optional location filter (e.g., 'San Francisco')
        filter_experience_level: Optional experience level filter (e.g., 'senior')
    
    Returns:
        Dictionary containing:
        - resume_id: Unique identifier for the resume
        - resume_name: Name of the candidate
        - categorized_jobs: List of matched job posts with scores
        - confidence_summary: Summary of confidence levels
        - processing_time: Time taken to process
    """
    try:
        result = await resume_tool.categorize_resume_to_posts(
            resume_path=resume_path,
            top_k=top_k,
            min_confidence=min_confidence,
            filter_location=filter_location,
            filter_experience_level=filter_experience_level
        )
        
        return {
            "success": True,
            "resume_id": result.resume_id,
            "resume_name": result.resume_name,
            "categorized_jobs": result.categorized_jobs,
            "confidence_summary": result.confidence_summary,
            "processing_time": result.processing_time,
            "message": f"Successfully categorized resume to {len(result.categorized_jobs)} job posts"
        }
    except Exception as e:
        logger.error(f"Error categorizing resume: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to categorize resume"
        }

@mcp.tool(
    description="Batch categorize multiple resumes to job posts"
)
async def batch_categorize_resumes(
    resume_paths: List[str],
    top_k: int = 5,
    parallel: bool = True
) -> Dict[str, Any]:
    """
    Categorize multiple resumes in batch.
    
    Args:
        resume_paths: List of paths to resume files
        top_k: Number of top matches per resume (default: 5)
        parallel: Process resumes in parallel (default: True)
    
    Returns:
        Dictionary containing results for each resume
    """
    try:
        results = await resume_tool.batch_categorize(
            resume_paths=resume_paths,
            top_k=top_k,
            parallel=parallel
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "resume_id": result.resume_id,
                "resume_name": result.resume_name,
                "top_matches": [
                    {
                        "job_title": job['job_title'],
                        "company": job['company'],
                        "score": job['final_score'],
                        "confidence": job['confidence']
                    }
                    for job in result.categorized_jobs[:3]  # Top 3 for summary
                ],
                "total_matches": len(result.categorized_jobs)
            })
        
        return {
            "success": True,
            "total_resumes": len(resume_paths),
            "processed": len(results),
            "results": formatted_results,
            "message": f"Successfully processed {len(results)} resumes"
        }
    except Exception as e:
        logger.error(f"Error in batch categorization: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to process batch"
        }

@mcp.tool(
    description="Train the categorizer model on historical matching data"
)
async def train_categorizer(
    match_history_file: str,
    min_score_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Train the categorizer on historical match data.
    
    Args:
        match_history_file: Path to JSON file containing historical matches
        min_score_threshold: Minimum score to consider as positive example (default: 0.7)
    
    Returns:
        Training summary including number of samples and success status
    """
    try:
        import json
        
        # Load historical data
        with open(match_history_file, 'r') as f:
            match_history = json.load(f)
        
        result = await resume_tool.train_on_historical_data(
            match_history=match_history,
            min_score_threshold=min_score_threshold
        )
        
        return {
            "success": result["success"],
            "training_samples": result.get("training_samples", 0),
            "positive_samples": result.get("positive_samples", 0),
            "model_saved": result.get("model_saved", False),
            "message": "Model training completed successfully" if result["success"] else "Training failed"
        }
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to train model"
        }

@mcp.tool(
    description="Update the ensemble model weights for semantic vs feature-based matching"
)
async def update_model_weights(
    semantic_weight: float = 0.6,
    feature_weight: float = 0.4
) -> Dict[str, Any]:
    """
    Update the weights used in the ensemble model.
    
    Args:
        semantic_weight: Weight for semantic similarity (0-1, default: 0.6)
        feature_weight: Weight for feature-based classification (0-1, default: 0.4)
    
    Returns:
        Updated weight configuration
    """
    try:
        result = await resume_tool.update_model_weights(
            semantic_weight=semantic_weight,
            feature_weight=feature_weight
        )
        
        return {
            "success": True,
            "semantic_weight": result["semantic_weight"],
            "feature_weight": result["feature_weight"],
            "message": "Model weights updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating weights: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to update weights"
        }

@mcp.tool(
    description="Get detailed explanation for why a resume matches a specific job"
)
async def explain_match(
    resume_path: str,
    job_id: str
) -> Dict[str, Any]:
    """
    Get a detailed explanation for why a resume matches a specific job post.
    
    Args:
        resume_path: Path to the resume file
        job_id: ID of the job post
    
    Returns:
        Detailed match explanation including skills, experience, and recommendations
    """
    try:
        explanation = await resume_tool.get_match_explanation(
            resume_path=resume_path,
            job_id=job_id
        )
        
        return {
            "success": True,
            "explanation": explanation,
            "message": "Match explanation generated successfully"
        }
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to generate explanation"
        }

@mcp.resource(
    uri="resume2post://status",
    name="Categorizer Status",
    description="Get the current status of the Resume2Post categorizer"
)
async def get_status() -> str:
    """Get current status of the categorizer"""
    try:
        status = {
            "model_loaded": resume_tool.categorizer.is_trained,
            "semantic_weight": resume_tool.categorizer.semantic_weight,
            "feature_weight": resume_tool.categorizer.feature_weight,
            "vector_store_active": True,  # Assume active if no exception
            "ready": True
        }
        
        import json
        return json.dumps(status, indent=2)
    except Exception as e:
        return f"Error getting status: {str(e)}"

@mcp.resource(
    uri="resume2post://help",
    name="Help Documentation",
    description="Get help documentation for using the Resume2Post categorizer"
)
async def get_help() -> str:
    """Get help documentation"""
    help_text = """
Resume2Post Categorizer - MCP Server Help

OVERVIEW:
This MCP server provides AI-powered resume categorization to specific job posts.
It uses a hybrid approach combining semantic similarity and feature-based classification.

MAIN FEATURES:
1. Single Resume Categorization
   - Categorize one resume to find best matching job posts
   - Filter by confidence level, location, and experience

2. Batch Processing
   - Process multiple resumes simultaneously
   - Parallel processing for efficiency

3. Model Training
   - Train on historical match data
   - Improve accuracy over time

4. Match Explanation
   - Get detailed explanations for matches
   - Understand why a resume fits a job

USAGE EXAMPLES:

1. Categorize a single resume:
   categorize_resume(
       resume_path="/path/to/resume.pdf",
       top_k=10,
       min_confidence="medium"
   )

2. Batch process resumes:
   batch_categorize_resumes(
       resume_paths=["/path/to/resume1.pdf", "/path/to/resume2.pdf"],
       top_k=5
   )

3. Train the model:
   train_categorizer(
       match_history_file="/path/to/history.json",
       min_score_threshold=0.7
   )

4. Update model weights:
   update_model_weights(
       semantic_weight=0.7,
       feature_weight=0.3
   )

CONFIDENCE LEVELS:
- high: Very strong match (>80% confidence)
- medium: Good match (60-80% confidence)  
- low: Potential match (<60% confidence)

For more information, see the project documentation.
"""
    return help_text

if __name__ == "__main__":
    import asyncio
    
    print("🚀 Starting Resume2Post MCP Server...")
    print("Server is ready to categorize resumes to job posts!")
    print("\nAvailable tools:")
    print("- categorize_resume: Categorize a single resume")
    print("- batch_categorize_resumes: Process multiple resumes")
    print("- train_categorizer: Train on historical data")
    print("- update_model_weights: Adjust model parameters")
    print("- explain_match: Get detailed match explanations")
    
    # Run the server
    asyncio.run(mcp.run())
"""
Async version of the main API with improved performance.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
from pydantic import BaseModel
import os
import aiofiles
from pathlib import Path
from datetime import datetime
import uuid
import asyncio

from agents.orchestrator_agent import OrchestratorAgent
from models.schemas import (
    Resume, JobPosition, MatchResult, BatchMatchRequest, 
    BatchMatchResponse, SalaryResearch
)
from tools.async_vector_store import get_async_vector_store
from tools.async_document_loaders import AsyncDocumentLoader, AsyncResumeLoader
from tools.async_web_scraper import get_async_scraper
from config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HR Resume Matcher API (Async)",
    description="High-performance async AI-powered resume matching system",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize async components
vector_store = None
document_loader = None
resume_loader = None
scraper = None
orchestrator = None

# Request/Response models
class SingleMatchRequest(BaseModel):
    resume_id: str
    position_id: str
    include_salary_research: bool = True
    include_aspiration_analysis: bool = True

class UploadResponse(BaseModel):
    id: str
    filename: str
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    vector_store_status: str

# Create upload directories
UPLOAD_DIR = Path("uploads")
RESUME_DIR = UPLOAD_DIR / "resumes"
POSITION_DIR = UPLOAD_DIR / "positions"

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize async components on startup"""
    global vector_store, document_loader, resume_loader, scraper, orchestrator
    
    # Create directories
    for dir_path in [UPLOAD_DIR, RESUME_DIR, POSITION_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    vector_store = await get_async_vector_store()
    document_loader = AsyncDocumentLoader()
    resume_loader = AsyncResumeLoader()
    scraper = await get_async_scraper()
    orchestrator = OrchestratorAgent()
    
    logger.info("Async components initialized successfully")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "HR Resume Matcher API (Async)",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "upload_resume": "/upload/resume",
            "upload_position": "/upload/position",
            "match_single": "/match/single",
            "match_batch": "/match/batch",
            "search_salaries": "/research/salary"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health asynchronously"""
    try:
        # Check Ollama connection asynchronously
        ollama_status = "healthy"
        try:
            # Test LLM connection (would need async version)
            test_response = await asyncio.get_event_loop().run_in_executor(
                None, orchestrator.llm.invoke, "test"
            )
            if not test_response:
                ollama_status = "unhealthy"
        except:
            ollama_status = "disconnected"
        
        # Check vector store
        vector_store_status = "healthy"
        try:
            # Test vector store connection
            await asyncio.get_event_loop().run_in_executor(
                None, orchestrator.vector_store.client.list_collections
            )
        except:
            vector_store_status = "unhealthy"
        
        overall_status = "healthy" if ollama_status == "healthy" and vector_store_status == "healthy" else "degraded"
        
        return HealthResponse(
            status=overall_status,
            ollama_status=ollama_status,
            vector_store_status=vector_store_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/upload/resume", response_model=UploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload and parse a resume asynchronously"""
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique ID
        resume_id = f"resume_{uuid.uuid4().hex[:8]}"
        file_path = RESUME_DIR / f"{resume_id}{file_ext}"
        
        # Save file asynchronously
        async with aiofiles.open(file_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)
        
        # Process resume in background
        background_tasks.add_task(process_resume_upload_async, resume_id, file_path)
        
        return UploadResponse(
            id=resume_id,
            filename=file.filename,
            status="processing",
            message="Resume uploaded successfully. Processing in background."
        )
        
    except Exception as e:
        logger.error(f"Error uploading resume: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/position", response_model=UploadResponse)
async def upload_position(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload and parse a job position asynchronously"""
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique ID
        position_id = f"pos_{uuid.uuid4().hex[:8]}"
        file_path = POSITION_DIR / f"{position_id}{file_ext}"
        
        # Save file asynchronously
        async with aiofiles.open(file_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)
        
        # Process position in background
        background_tasks.add_task(process_position_upload_async, position_id, file_path)
        
        return UploadResponse(
            id=position_id,
            filename=file.filename,
            status="processing",
            message="Position uploaded successfully. Processing in background."
        )
        
    except Exception as e:
        logger.error(f"Error uploading position: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match/single", response_model=dict)
async def match_single(request: SingleMatchRequest):
    """Perform single resume-position matching asynchronously"""
    try:
        # Get file paths
        resume_path = await find_file_by_id_async(RESUME_DIR, request.resume_id)
        position_path = await find_file_by_id_async(POSITION_DIR, request.position_id)
        
        if not resume_path:
            raise HTTPException(status_code=404, detail="Resume not found")
        if not position_path:
            raise HTTPException(status_code=404, detail="Position not found")
        
        # Load documents asynchronously
        resume_doc = await document_loader.load_document(resume_path)
        position_doc = await document_loader.load_document(position_path)
        
        # Extract text content
        resume_text = extract_text_from_doc(resume_doc)
        position_text = extract_text_from_doc(position_doc)
        
        # Perform vector similarity search
        similar_positions = await vector_store.search_similar_positions(resume_text, k=1)
        
        # Calculate match score
        match_score = similar_positions[0]["similarity_score"] if similar_positions else 0.0
        
        result = {
            "success": True,
            "match": {
                "resume_id": request.resume_id,
                "position_id": request.position_id,
                "overall_score": match_score,
                "match_percentage": round(match_score * 100, 2)
            }
        }
        
        # Add salary research if requested
        if request.include_salary_research:
            salary_data = await scraper.aggregate_salary_data(
                job_title=position_doc.get("metadata", {}).get("title", "Unknown"),
                location=position_doc.get("metadata", {}).get("location", "Unknown")
            )
            result["salary_research"] = salary_data
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in single matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match/batch", response_model=BatchMatchResponse)
async def match_batch(request: BatchMatchRequest):
    """Perform batch matching for multiple resumes and positions asynchronously"""
    try:
        matches = []
        
        # Create tasks for concurrent processing
        tasks = []
        for resume_id in request.resume_ids:
            for position_id in request.position_ids:
                task = match_single_pair(resume_id, position_id)
                tasks.append(task)
        
        # Execute all matches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, dict) and result.get("success"):
                matches.append(result["match"])
            else:
                logger.error(f"Batch match error: {result}")
        
        # Sort by score
        matches.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Apply limit
        if request.top_k:
            matches = matches[:request.top_k]
        
        return BatchMatchResponse(
            matches=matches,
            total_comparisons=len(request.resume_ids) * len(request.position_ids),
            processing_time=0.0  # Would calculate actual time
        )
        
    except Exception as e:
        logger.error(f"Error in batch matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/salary", response_model=dict)
async def research_salary(
    position_title: str,
    location: str,
    experience_years: Optional[int] = None,
    company: Optional[str] = None
):
    """Research salary for a position asynchronously"""
    try:
        # Use async scraper
        result = await scraper.aggregate_salary_data(
            job_title=position_title,
            location=location,
            experience_years=experience_years,
            company=company
        )
        
        return {
            "success": True,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error in salary research: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def process_resume_upload_async(resume_id: str, file_path: Path):
    """Process uploaded resume in background asynchronously"""
    try:
        # Load resume asynchronously
        resume_data = await resume_loader.load_resume(file_path)
        
        # Parse resume data (would need async parser)
        parsed_resume = {
            "id": resume_id,
            "name": extract_name_from_resume(resume_data),
            "email": extract_email_from_resume(resume_data),
            "total_experience_years": extract_experience_years(resume_data),
            "skills": extract_skills(resume_data),
            "location": extract_location(resume_data)
        }
        
        # Store in vector database asynchronously
        await vector_store.add_resume(parsed_resume)
        logger.info(f"Successfully processed resume {resume_id}")
        
    except Exception as e:
        logger.error(f"Error processing resume {resume_id}: {e}")

async def process_position_upload_async(position_id: str, file_path: Path):
    """Process uploaded position in background asynchronously"""
    try:
        # Load position asynchronously
        position_doc = await document_loader.load_document(file_path)
        
        # Parse position data (would need async parser)
        parsed_position = {
            "id": position_id,
            "title": extract_title_from_position(position_doc),
            "department": "Unknown",
            "location": extract_location_from_position(position_doc),
            "description": extract_text_from_doc(position_doc),
            "experience_level": "Mid-Senior",
            "min_experience_years": 3,
            "required_skills": extract_skills_from_position(position_doc)
        }
        
        # Store in vector database asynchronously
        await vector_store.add_position(parsed_position)
        logger.info(f"Successfully processed position {position_id}")
        
    except Exception as e:
        logger.error(f"Error processing position {position_id}: {e}")

# Utility functions
async def find_file_by_id_async(directory: Path, file_id: str) -> Optional[Path]:
    """Find file by ID prefix in directory asynchronously"""
    try:
        # List directory contents asynchronously
        files = await asyncio.get_event_loop().run_in_executor(
            None, list, directory.glob(f"{file_id}*")
        )
        return files[0] if files else None
    except Exception:
        return None

async def match_single_pair(resume_id: str, position_id: str) -> dict:
    """Match a single resume-position pair"""
    try:
        request = SingleMatchRequest(
            resume_id=resume_id,
            position_id=position_id,
            include_salary_research=False,
            include_aspiration_analysis=False
        )
        return await match_single(request)
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_text_from_doc(doc: dict) -> str:
    """Extract text content from document data"""
    if doc["type"] == "pdf":
        return "\n".join([page["text"] for page in doc["content"]])
    elif doc["type"] == "docx":
        return doc["content"]["text"]
    elif doc["type"] in ["text", "txt"]:
        return doc["content"]
    else:
        return str(doc.get("content", ""))

# Placeholder extraction functions (would be implemented properly)
def extract_name_from_resume(data: dict) -> str:
    return "John Doe"  # Placeholder

def extract_email_from_resume(data: dict) -> str:
    return "john.doe@example.com"  # Placeholder

def extract_experience_years(data: dict) -> int:
    return 5  # Placeholder

def extract_skills(data: dict) -> list:
    return [{"name": "Python", "level": "Expert"}]  # Placeholder

def extract_location(data: dict) -> str:
    return "San Francisco, CA"  # Placeholder

def extract_title_from_position(data: dict) -> str:
    return "Software Engineer"  # Placeholder

def extract_location_from_position(data: dict) -> str:
    return "San Francisco, CA"  # Placeholder

def extract_skills_from_position(data: dict) -> list:
    return ["Python", "FastAPI", "Async"]  # Placeholder

if __name__ == "__main__":
    uvicorn.run(
        "async_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
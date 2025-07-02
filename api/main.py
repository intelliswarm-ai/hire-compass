from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
from pydantic import BaseModel
import os
import shutil
from datetime import datetime
import uuid

from agents.orchestrator_agent import OrchestratorAgent
from models.schemas import (
    Resume, JobPosition, MatchResult, BatchMatchRequest, 
    BatchMatchResponse, SalaryResearch
)
from config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HR Resume Matcher API",
    description="AI-powered resume matching system with multi-agent support",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = OrchestratorAgent()

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
UPLOAD_DIR = "uploads"
RESUME_DIR = os.path.join(UPLOAD_DIR, "resumes")
POSITION_DIR = os.path.join(UPLOAD_DIR, "positions")

for dir_path in [UPLOAD_DIR, RESUME_DIR, POSITION_DIR]:
    os.makedirs(dir_path, exist_ok=True)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "HR Resume Matcher API",
        "version": "1.0.0",
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
    """Check system health"""
    try:
        # Check Ollama connection
        ollama_status = "healthy"
        try:
            # Test LLM connection
            test_response = orchestrator.llm.invoke("test")
            if not test_response:
                ollama_status = "unhealthy"
        except:
            ollama_status = "disconnected"
        
        # Check vector store
        vector_store_status = "healthy"
        try:
            # Test vector store connection
            orchestrator.vector_store.client.list_collections()
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
    """Upload and parse a resume"""
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
        file_path = os.path.join(RESUME_DIR, f"{resume_id}{file_ext}")
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process resume in background
        background_tasks.add_task(process_resume_upload, resume_id, file_path)
        
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
    """Upload and parse a job position"""
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
        file_path = os.path.join(POSITION_DIR, f"{position_id}{file_ext}")
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process position in background
        background_tasks.add_task(process_position_upload, position_id, file_path)
        
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
    """Perform single resume-position matching"""
    try:
        # Get file paths
        resume_path = find_file_by_id(RESUME_DIR, request.resume_id)
        position_path = find_file_by_id(POSITION_DIR, request.position_id)
        
        if not resume_path:
            raise HTTPException(status_code=404, detail="Resume not found")
        if not position_path:
            raise HTTPException(status_code=404, detail="Position not found")
        
        # Perform matching
        result = orchestrator.process_single_match(
            resume_path=resume_path,
            position_path=position_path,
            include_salary=request.include_salary_research,
            include_aspirations=request.include_aspiration_analysis
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Matching failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in single matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match/batch", response_model=BatchMatchResponse)
async def match_batch(request: BatchMatchRequest):
    """Perform batch matching for multiple resumes and positions"""
    try:
        # Process batch request
        result = orchestrator.process({
            "batch_request": request.dict()
        })
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Batch matching failed"))
        
        return BatchMatchResponse(**result["batch_response"])
        
    except Exception as e:
        logger.error(f"Error in batch matching: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/salary", response_model=dict)
async def research_salary(
    position_title: str,
    location: str,
    experience_years: Optional[int] = None
):
    """Research salary for a position"""
    try:
        result = orchestrator.salary_agent.process({
            "position_title": position_title,
            "location": location,
            "experience_years": experience_years
        })
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Salary research failed"))
        
        return result["result"]
        
    except Exception as e:
        logger.error(f"Error in salary research: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def process_resume_upload(resume_id: str, file_path: str):
    """Process uploaded resume in background"""
    try:
        result = orchestrator.resume_parser.process({
            "file_path": file_path,
            "resume_id": resume_id
        })
        
        if result["success"]:
            # Store in vector database
            orchestrator.vector_store.add_resume(result["resume"])
            logger.info(f"Successfully processed resume {resume_id}")
        else:
            logger.error(f"Failed to process resume {resume_id}: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error processing resume {resume_id}: {e}")

async def process_position_upload(position_id: str, file_path: str):
    """Process uploaded position in background"""
    try:
        result = orchestrator.job_parser.process({
            "file_path": file_path,
            "position_id": position_id
        })
        
        if result["success"]:
            # Store in vector database
            orchestrator.vector_store.add_position(result["position"])
            logger.info(f"Successfully processed position {position_id}")
        else:
            logger.error(f"Failed to process position {position_id}: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Error processing position {position_id}: {e}")

# Utility functions
def find_file_by_id(directory: str, file_id: str) -> Optional[str]:
    """Find file by ID prefix in directory"""
    for filename in os.listdir(directory):
        if filename.startswith(file_id):
            return os.path.join(directory, filename)
    return None

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
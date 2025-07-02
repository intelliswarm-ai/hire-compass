import os
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    vector_store_collection: str = os.getenv("VECTOR_STORE_COLLECTION", "hr_resume_matcher")
    max_concurrent_agents: int = int(os.getenv("MAX_CONCURRENT_AGENTS", "5"))
    salary_research_timeout: int = int(os.getenv("SALARY_RESEARCH_TIMEOUT", "30"))
    web_driver_path: str = os.getenv("WEB_DRIVER_PATH", "/usr/local/bin/chromedriver")
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "nomic-embed-text"
    
    class Config:
        env_file = ".env"

settings = Settings()
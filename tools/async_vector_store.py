"""
Async version of vector store manager with improved performance.
"""

from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from config import settings as app_settings
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AsyncVectorStoreManager:
    """Async vector storage manager for resumes and job positions"""
    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            base_url=app_settings.ollama_base_url,
            model=app_settings.embedding_model
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=app_settings.chroma_persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Create or get collections
        self.resume_collection = self._get_or_create_collection("resumes")
        self.position_collection = self._get_or_create_collection("positions")
        
        # Initialize Langchain Chroma wrappers
        self.resume_store = Chroma(
            client=self.client,
            collection_name="resumes",
            embedding_function=self.embeddings
        )
        
        self.position_store = Chroma(
            client=self.client,
            collection_name="positions",
            embedding_function=self.embeddings
        )
        
        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Async vector store manager initialized")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
    
    async def add_resume(self, resume_data: Dict[str, Any]) -> str:
        """Add a resume to the vector store asynchronously"""
        try:
            # Create document from resume
            doc_text = self._create_resume_text(resume_data)
            
            # Run embedding and storage in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._add_resume_sync,
                doc_text,
                resume_data
            )
            
            logger.info(f"Added resume {resume_data['id']} to vector store")
            return resume_data["id"]
            
        except Exception as e:
            logger.error(f"Error adding resume to vector store: {str(e)}")
            raise
    
    def _add_resume_sync(self, doc_text: str, resume_data: Dict[str, Any]):
        """Synchronous helper for adding resume"""
        return self.resume_store.add_texts(
            texts=[doc_text],
            metadatas=[{
                "id": resume_data["id"],
                "name": resume_data["name"],
                "email": resume_data["email"],
                "total_experience_years": resume_data["total_experience_years"],
                "skills": ",".join([s["name"] for s in resume_data.get("skills", [])]),
                "location": resume_data.get("location", ""),
                "expected_salary": resume_data.get("expected_salary", 0)
            }],
            ids=[resume_data["id"]]
        )
    
    async def add_position(self, position_data: Dict[str, Any]) -> str:
        """Add a job position to the vector store asynchronously"""
        try:
            # Create document from position
            doc_text = self._create_position_text(position_data)
            
            # Run embedding and storage in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._add_position_sync,
                doc_text,
                position_data
            )
            
            logger.info(f"Added position {position_data['id']} to vector store")
            return position_data["id"]
            
        except Exception as e:
            logger.error(f"Error adding position to vector store: {str(e)}")
            raise
    
    def _add_position_sync(self, doc_text: str, position_data: Dict[str, Any]):
        """Synchronous helper for adding position"""
        return self.position_store.add_texts(
            texts=[doc_text],
            metadatas=[{
                "id": position_data["id"],
                "title": position_data["title"],
                "department": position_data["department"],
                "location": position_data["location"],
                "experience_level": position_data["experience_level"],
                "min_experience_years": position_data["min_experience_years"],
                "required_skills": ",".join(position_data.get("required_skills", [])),
                "salary_range_min": position_data.get("salary_range_min", 0),
                "salary_range_max": position_data.get("salary_range_max", 0)
            }],
            ids=[position_data["id"]]
        )
    
    async def search_similar_positions(self, resume_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for positions similar to a resume asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self._executor,
                self.position_store.similarity_search_with_score,
                resume_text,
                k
            )
            
            return [
                {
                    "position_id": doc.metadata["id"],
                    "title": doc.metadata["title"],
                    "similarity_score": 1 - score,  # Convert distance to similarity
                    "metadata": doc.metadata
                }
                for doc, score in results
            ]
            
        except Exception as e:
            logger.error(f"Error searching positions: {str(e)}")
            return []
    
    async def search_similar_resumes(self, position_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for resumes similar to a position asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self._executor,
                self.resume_store.similarity_search_with_score,
                position_text,
                k
            )
            
            return [
                {
                    "resume_id": doc.metadata["id"],
                    "name": doc.metadata["name"],
                    "similarity_score": 1 - score,  # Convert distance to similarity
                    "metadata": doc.metadata
                }
                for doc, score in results
            ]
            
        except Exception as e:
            logger.error(f"Error searching resumes: {str(e)}")
            return []
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.embeddings.embed_documents,
            texts
        )
    
    async def batch_add_resumes(self, resumes: List[Dict[str, Any]], batch_size: int = 10) -> List[str]:
        """Add multiple resumes in batches for better performance"""
        results = []
        
        for i in range(0, len(resumes), batch_size):
            batch = resumes[i:i + batch_size]
            tasks = [self.add_resume(resume) for resume in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch processing: {result}")
                else:
                    results.append(result)
        
        return results
    
    async def batch_add_positions(self, positions: List[Dict[str, Any]], batch_size: int = 10) -> List[str]:
        """Add multiple positions in batches for better performance"""
        results = []
        
        for i in range(0, len(positions), batch_size):
            batch = positions[i:i + batch_size]
            tasks = [self.add_position(position) for position in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in batch processing: {result}")
                else:
                    results.append(result)
        
        return results
    
    def _create_resume_text(self, resume_data: Dict[str, Any]) -> str:
        """Create searchable text representation of resume"""
        parts = [
            f"Name: {resume_data['name']}",
            f"Summary: {resume_data.get('summary', '')}",
            f"Total Experience: {resume_data['total_experience_years']} years",
            f"Location: {resume_data.get('location', '')}",
            f"Skills: {', '.join([s['name'] for s in resume_data.get('skills', [])])}",
            f"Education: {', '.join([f\"{e['degree']} in {e['field']}\" for e in resume_data.get('education', [])])}",
            f"Experience: {' | '.join([f\"{e['position']} at {e['company']}\" for e in resume_data.get('experience', [])])}"
        ]
        
        return "\n".join(parts)
    
    def _create_position_text(self, position_data: Dict[str, Any]) -> str:
        """Create searchable text representation of position"""
        parts = [
            f"Title: {position_data['title']}",
            f"Department: {position_data['department']}",
            f"Location: {position_data['location']}",
            f"Description: {position_data['description']}",
            f"Experience Level: {position_data['experience_level']}",
            f"Min Experience: {position_data['min_experience_years']} years",
            f"Required Skills: {', '.join(position_data.get('required_skills', []))}",
            f"Preferred Skills: {', '.join(position_data.get('preferred_skills', []))}",
            f"Responsibilities: {' | '.join(position_data.get('responsibilities', []))}",
            f"Requirements: {' | '.join(position_data.get('requirements', []))}"
        ]
        
        return "\n".join(parts)
    
    async def delete_resume(self, resume_id: str):
        """Delete a resume from the vector store asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self.resume_collection.delete,
                [resume_id]
            )
            logger.info(f"Deleted resume {resume_id} from vector store")
        except Exception as e:
            logger.error(f"Error deleting resume: {str(e)}")
    
    async def delete_position(self, position_id: str):
        """Delete a position from the vector store asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self.position_collection.delete,
                [position_id]
            )
            logger.info(f"Deleted position {position_id} from vector store")
        except Exception as e:
            logger.error(f"Error deleting position: {str(e)}")
    
    async def clear_all(self):
        """Clear all data from vector stores asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._executor,
                self._clear_all_sync
            )
            logger.info("Cleared all vector stores")
        except Exception as e:
            logger.error(f"Error clearing vector stores: {str(e)}")
    
    def _clear_all_sync(self):
        """Synchronous helper for clearing all collections"""
        self.client.delete_collection("resumes")
        self.client.delete_collection("positions")
        self.resume_collection = self._get_or_create_collection("resumes")
        self.position_collection = self._get_or_create_collection("positions")
    
    def __del__(self):
        """Cleanup thread pool executor"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Singleton instance
_async_vector_store = None


async def get_async_vector_store() -> AsyncVectorStoreManager:
    """Get or create async vector store singleton"""
    global _async_vector_store
    if _async_vector_store is None:
        _async_vector_store = AsyncVectorStoreManager()
    return _async_vector_store
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from config import settings as app_settings
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector storage for resumes and job positions"""
    
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
        
        logger.info("Vector store manager initialized")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_resume(self, resume_data: Dict[str, Any]) -> str:
        """Add a resume to the vector store"""
        try:
            # Create document from resume
            doc_text = self._create_resume_text(resume_data)
            
            # Store in vector database
            result = self.resume_store.add_texts(
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
            
            logger.info(f"Added resume {resume_data['id']} to vector store")
            return resume_data["id"]
            
        except Exception as e:
            logger.error(f"Error adding resume to vector store: {str(e)}")
            raise
    
    def add_position(self, position_data: Dict[str, Any]) -> str:
        """Add a job position to the vector store"""
        try:
            # Create document from position
            doc_text = self._create_position_text(position_data)
            
            # Store in vector database
            result = self.position_store.add_texts(
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
            
            logger.info(f"Added position {position_data['id']} to vector store")
            return position_data["id"]
            
        except Exception as e:
            logger.error(f"Error adding position to vector store: {str(e)}")
            raise
    
    def search_similar_positions(self, resume_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for positions similar to a resume"""
        try:
            results = self.position_store.similarity_search_with_score(
                query=resume_text,
                k=k
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
    
    def search_similar_resumes(self, position_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for resumes similar to a position"""
        try:
            results = self.resume_store.similarity_search_with_score(
                query=position_text,
                k=k
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
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        return self.embeddings.embed_documents(texts)
    
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
    
    def delete_resume(self, resume_id: str):
        """Delete a resume from the vector store"""
        try:
            self.resume_collection.delete(ids=[resume_id])
            logger.info(f"Deleted resume {resume_id} from vector store")
        except Exception as e:
            logger.error(f"Error deleting resume: {str(e)}")
    
    def delete_position(self, position_id: str):
        """Delete a position from the vector store"""
        try:
            self.position_collection.delete(ids=[position_id])
            logger.info(f"Deleted position {position_id} from vector store")
        except Exception as e:
            logger.error(f"Error deleting position: {str(e)}")
    
    def clear_all(self):
        """Clear all data from vector stores"""
        try:
            self.client.delete_collection("resumes")
            self.client.delete_collection("positions")
            self.resume_collection = self._get_or_create_collection("resumes")
            self.position_collection = self._get_or_create_collection("positions")
            logger.info("Cleared all vector stores")
        except Exception as e:
            logger.error(f"Error clearing vector stores: {str(e)}")
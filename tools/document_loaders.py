import os
from typing import List, Union
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class ResumeLoader:
    """Unified loader for various resume formats"""
    
    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        """Load a document based on its file extension"""
        _, ext = os.path.splitext(file_path.lower())
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif ext in ['.txt', '.text']:
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} pages from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_batch(file_paths: List[str]) -> List[Document]:
        """Load multiple documents"""
        all_documents = []
        for file_path in file_paths:
            try:
                documents = ResumeLoader.load_document(file_path)
                for doc in documents:
                    doc.metadata['source_file'] = file_path
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                continue
        
        return all_documents
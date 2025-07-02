from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from config import settings

class ResumeSplitter:
    """Custom text splitter optimized for resume content"""
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", ";", ",", " ", ""],
            length_function=len
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks while preserving metadata"""
        return self.splitter.split_documents(documents)
    
    def split_text(self, text: str) -> List[str]:
        """Split plain text into chunks"""
        return self.splitter.split_text(text)
import re
import string
import spacy
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ResumeTextPreprocessor:
    """Advanced text preprocessing for resume categorization"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,5}[-\s\.]?[0-9]{1,5}', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from text"""
        # Common technical skills patterns
        skill_patterns = [
            r'\b(python|java|javascript|c\+\+|c#|ruby|go|rust|swift|kotlin|scala|r|matlab)\b',
            r'\b(react|angular|vue|django|flask|spring|node\.?js|express|fastapi)\b',
            r'\b(aws|azure|gcp|docker|kubernetes|terraform|ansible|jenkins)\b',
            r'\b(sql|nosql|mongodb|postgresql|mysql|redis|elasticsearch)\b',
            r'\b(machine learning|deep learning|nlp|computer vision|ai|ml|dl)\b',
            r'\b(git|github|gitlab|bitbucket|svn|ci/cd|devops)\b',
            r'\b(agile|scrum|kanban|jira|confluence)\b',
            r'\b(tensorflow|pytorch|keras|scikit-learn|pandas|numpy)\b'
        ]
        
        skills = []
        text_lower = text.lower()
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text_lower)
            skills.extend(matches)
        
        # Remove duplicates and return
        return list(set(skills))
    
    def extract_experience_years(self, text: str) -> float:
        """Extract total years of experience from text"""
        # Look for patterns like "X years of experience"
        patterns = [
            r'(\d+\.?\d*)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?\s*(\d+\.?\d*)\+?\s*years?',
            r'(\d+\.?\d*)\+?\s*years?\s*in\s*the\s*industry',
            r'(\d+\.?\d*)\+?\s*years?\s*professional'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        
        # Fallback: count work experience entries
        work_sections = re.findall(r'(\d{4})\s*[-–]\s*(\d{4}|present)', text.lower())
        if work_sections:
            total_years = 0
            for start, end in work_sections:
                try:
                    start_year = int(start)
                    end_year = 2024 if end == 'present' else int(end)
                    total_years += (end_year - start_year)
                except:
                    continue
            return float(total_years)
        
        return 0.0
    
    def extract_education_level(self, text: str) -> str:
        """Extract highest education level"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['phd', 'ph.d', 'doctorate']):
            return 'phd'
        elif any(term in text_lower for term in ['master', 'mba', 'm.s.', 'ms ', 'ma ', 'm.a.']):
            return 'masters'
        elif any(term in text_lower for term in ['bachelor', 'b.s.', 'bs ', 'ba ', 'b.a.', 'undergraduate']):
            return 'bachelors'
        elif any(term in text_lower for term in ['associate', 'diploma']):
            return 'associate'
        else:
            return 'unknown'
    
    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from resume"""
        sections = {
            'summary': '',
            'experience': '',
            'education': '',
            'skills': '',
            'projects': ''
        }
        
        # Section headers patterns
        section_patterns = {
            'summary': r'(professional summary|summary|objective|profile)(.*?)(?=experience|education|skills|projects|$)',
            'experience': r'(work experience|experience|employment|professional experience)(.*?)(?=education|skills|projects|$)',
            'education': r'(education|academic|qualification)(.*?)(?=skills|projects|experience|$)',
            'skills': r'(skills|technical skills|competencies)(.*?)(?=projects|experience|education|$)',
            'projects': r'(projects|portfolio)(.*?)(?=education|skills|experience|$)'
        }
        
        text_lower = text.lower()
        
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section] = match.group(2).strip()[:1000]  # Limit section length
        
        return sections
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy"""
        entities = {
            'organizations': [],
            'locations': [],
            'persons': [],
            'dates': []
        }
        
        if self.nlp:
            doc = self.nlp(text[:1000000])  # Limit text length for processing
            
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities['organizations'].append(ent.text)
                elif ent.label_ in ["GPE", "LOC"]:
                    entities['locations'].append(ent.text)
                elif ent.label_ == "PERSON":
                    entities['persons'].append(ent.text)
                elif ent.label_ == "DATE":
                    entities['dates'].append(ent.text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def preprocess_resume(self, text: str) -> Dict[str, Any]:
        """Complete preprocessing pipeline for resume"""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract features
        features = {
            'cleaned_text': cleaned_text,
            'skills': self.extract_skills(text),
            'experience_years': self.extract_experience_years(text),
            'education_level': self.extract_education_level(text),
            'sections': self.extract_key_sections(text),
            'entities': self.extract_entities(text),
            'text_length': len(cleaned_text),
            'word_count': len(cleaned_text.split())
        }
        
        return features
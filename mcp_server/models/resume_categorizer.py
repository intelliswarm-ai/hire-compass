import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import logging
from datetime import datetime

from tools.vector_store import VectorStoreManager
from mcp_server.utils.text_preprocessor import ResumeTextPreprocessor

logger = logging.getLogger(__name__)

class Resume2PostCategorizer:
    """
    Categorizes resumes to specific job posts using hybrid approach:
    1. Semantic similarity using sentence embeddings
    2. Feature-based classification using Random Forest
    3. Weighted ensemble of both approaches
    """
    
    def __init__(self, model_path: str = None):
        self.preprocessor = ResumeTextPreprocessor()
        self.vector_store = VectorStoreManager()
        
        # Initialize sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize feature-based classifier
        self.feature_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Model storage path
        self.model_path = model_path or "mcp_server/models/saved"
        os.makedirs(self.model_path, exist_ok=True)
        
        # Feature weights for ensemble
        self.semantic_weight = 0.6
        self.feature_weight = 0.4
    
    def extract_features(self, resume_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from preprocessed resume"""
        features = []
        
        # Experience years (normalized)
        features.append(resume_data.get('experience_years', 0) / 30.0)
        
        # Education level encoding
        education_map = {
            'unknown': 0, 'associate': 1, 'bachelors': 2, 
            'masters': 3, 'phd': 4
        }
        features.append(education_map.get(resume_data.get('education_level', 'unknown'), 0) / 4.0)
        
        # Skills count (normalized)
        features.append(len(resume_data.get('skills', [])) / 50.0)
        
        # Text statistics
        features.append(resume_data.get('word_count', 0) / 1000.0)
        
        # Section presence (binary features)
        sections = resume_data.get('sections', {})
        for section in ['summary', 'experience', 'education', 'skills', 'projects']:
            features.append(1.0 if sections.get(section, '') else 0.0)
        
        # Entity counts (normalized)
        entities = resume_data.get('entities', {})
        features.append(len(entities.get('organizations', [])) / 20.0)
        features.append(len(entities.get('locations', [])) / 10.0)
        
        return np.array(features)
    
    def create_resume_embedding(self, resume_text: str) -> np.ndarray:
        """Create semantic embedding for resume"""
        # Use sentence transformer to create embedding
        embedding = self.sentence_model.encode(resume_text, show_progress_bar=False)
        return embedding
    
    def create_job_embedding(self, job_data: Dict[str, Any]) -> np.ndarray:
        """Create semantic embedding for job post"""
        # Combine relevant job fields
        job_text = f"{job_data.get('title', '')} {job_data.get('description', '')} " \
                  f"{' '.join(job_data.get('required_skills', []))} " \
                  f"{' '.join(job_data.get('responsibilities', []))}"
        
        embedding = self.sentence_model.encode(job_text, show_progress_bar=False)
        return embedding
    
    def train(self, training_data: List[Tuple[Dict, Dict, float]]):
        """
        Train the categorizer on historical matches
        training_data: List of (resume_data, job_data, match_score) tuples
        """
        if not training_data:
            logger.warning("No training data provided")
            return
        
        logger.info(f"Training on {len(training_data)} examples")
        
        # Prepare training features and labels
        X_features = []
        X_embeddings = []
        y_labels = []
        
        for resume_data, job_data, match_score in training_data:
            # Preprocess resume
            preprocessed = self.preprocessor.preprocess_resume(resume_data.get('raw_text', ''))
            
            # Extract features
            features = self.extract_features(preprocessed)
            X_features.append(features)
            
            # Create embeddings
            resume_emb = self.create_resume_embedding(preprocessed['cleaned_text'])
            job_emb = self.create_job_embedding(job_data)
            
            # Combine embeddings (concatenate or similarity score)
            similarity = cosine_similarity([resume_emb], [job_emb])[0][0]
            X_embeddings.append(similarity)
            
            # Binary classification: good match (>0.7) or not
            y_labels.append(1 if match_score > 0.7 else 0)
        
        # Scale features
        X_features = np.array(X_features)
        X_features_scaled = self.scaler.fit_transform(X_features)
        
        # Add embedding similarity as additional feature
        X_combined = np.column_stack([X_features_scaled, X_embeddings])
        
        # Train classifier
        self.feature_classifier.fit(X_combined, y_labels)
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        logger.info("Training completed")
    
    def categorize_resume(self, resume_path: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Categorize a resume to top K job posts
        Returns list of job posts with confidence scores
        """
        # Load and preprocess resume
        from tools.document_loaders import ResumeLoader
        documents = ResumeLoader.load_document(resume_path)
        raw_text = "\n".join([doc.page_content for doc in documents])
        
        preprocessed = self.preprocessor.preprocess_resume(raw_text)
        
        # Get all active job positions from vector store
        # For now, we'll use vector similarity search
        job_matches = self.vector_store.search_similar_positions(
            resume_text=preprocessed['cleaned_text'],
            k=top_k * 2  # Get more candidates for filtering
        )
        
        # Score each job match
        scored_matches = []
        
        for job_match in job_matches:
            # Calculate semantic similarity
            resume_emb = self.create_resume_embedding(preprocessed['cleaned_text'])
            
            # Get job data (in production, would fetch from database)
            job_data = job_match['metadata']
            job_emb = self.create_job_embedding(job_data)
            
            semantic_score = cosine_similarity([resume_emb], [job_emb])[0][0]
            
            # Calculate feature-based score if model is trained
            feature_score = 0.5  # Default neutral score
            
            if self.is_trained:
                features = self.extract_features(preprocessed)
                features_scaled = self.scaler.transform([features])
                combined_features = np.column_stack([features_scaled, [semantic_score]])
                
                # Get probability of good match
                feature_score = self.feature_classifier.predict_proba(combined_features)[0][1]
            
            # Ensemble score
            final_score = (self.semantic_weight * semantic_score + 
                          self.feature_weight * feature_score)
            
            scored_matches.append({
                'job_id': job_match['position_id'],
                'job_title': job_data.get('title', 'Unknown'),
                'company': job_data.get('department', 'Unknown'),
                'location': job_data.get('location', 'Unknown'),
                'semantic_score': semantic_score,
                'feature_score': feature_score,
                'final_score': final_score,
                'confidence': self._calculate_confidence(semantic_score, feature_score),
                'match_reasons': self._generate_match_reasons(preprocessed, job_data)
            })
        
        # Sort by final score and return top K
        scored_matches.sort(key=lambda x: x['final_score'], reverse=True)
        
        return scored_matches[:top_k]
    
    def batch_categorize(self, resume_paths: List[str], 
                        job_ids: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Categorize multiple resumes in batch
        Returns mapping of resume_path to categorized jobs
        """
        results = {}
        
        for resume_path in resume_paths:
            try:
                categorized = self.categorize_resume(resume_path)
                results[resume_path] = categorized
            except Exception as e:
                logger.error(f"Error categorizing {resume_path}: {e}")
                results[resume_path] = []
        
        return results
    
    def _calculate_confidence(self, semantic_score: float, feature_score: float) -> str:
        """Calculate confidence level for the match"""
        avg_score = (semantic_score + feature_score) / 2
        
        if avg_score > 0.8:
            return "high"
        elif avg_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def _generate_match_reasons(self, resume_data: Dict, job_data: Dict) -> List[str]:
        """Generate reasons why resume matches the job"""
        reasons = []
        
        # Skill matches
        resume_skills = set([s.lower() for s in resume_data.get('skills', [])])
        required_skills = set([s.lower() for s in job_data.get('required_skills', '').split(',')])
        
        matched_skills = resume_skills.intersection(required_skills)
        if matched_skills:
            reasons.append(f"Matching skills: {', '.join(list(matched_skills)[:5])}")
        
        # Experience match
        resume_years = resume_data.get('experience_years', 0)
        min_years = job_data.get('min_experience_years', 0)
        
        if resume_years >= min_years:
            reasons.append(f"Experience requirement met ({resume_years} years)")
        
        # Location match
        resume_location = resume_data.get('entities', {}).get('locations', [])
        job_location = job_data.get('location', '')
        
        if job_location and any(loc in job_location for loc in resume_location):
            reasons.append("Location match")
        
        return reasons
    
    def save_model(self):
        """Save trained model and scaler"""
        if self.is_trained:
            joblib.dump(self.feature_classifier, 
                       os.path.join(self.model_path, 'classifier.pkl'))
            joblib.dump(self.scaler, 
                       os.path.join(self.model_path, 'scaler.pkl'))
            logger.info("Model saved successfully")
    
    def load_model(self):
        """Load pre-trained model and scaler"""
        classifier_path = os.path.join(self.model_path, 'classifier.pkl')
        scaler_path = os.path.join(self.model_path, 'scaler.pkl')
        
        if os.path.exists(classifier_path) and os.path.exists(scaler_path):
            self.feature_classifier = joblib.load(classifier_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            logger.info("Model loaded successfully")
        else:
            logger.warning("No saved model found")
    
    def update_weights(self, semantic_weight: float, feature_weight: float):
        """Update ensemble weights"""
        if semantic_weight + feature_weight != 1.0:
            # Normalize weights
            total = semantic_weight + feature_weight
            semantic_weight /= total
            feature_weight /= total
        
        self.semantic_weight = semantic_weight
        self.feature_weight = feature_weight
        
        logger.info(f"Updated weights - Semantic: {self.semantic_weight:.2f}, "
                   f"Feature: {self.feature_weight:.2f}")
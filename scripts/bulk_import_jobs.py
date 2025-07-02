#!/usr/bin/env python
"""
Bulk import job positions into the HR Resume Matcher system
Supports multiple formats: JSON, CSV, API endpoints
"""

import json
import csv
import os
import sys
import time
import argparse
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.job_parser_agent import JobParserAgent
from tools.vector_store import VectorStoreManager
from models.schemas import JobPosition

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobBulkImporter:
    """Bulk import jobs from various sources"""
    
    def __init__(self):
        self.job_parser = JobParserAgent()
        self.vector_store = VectorStoreManager()
        self.imported_count = 0
        self.failed_count = 0
        
    def import_from_json(self, json_file: str) -> Dict[str, Any]:
        """Import jobs from JSON file"""
        logger.info(f"Importing jobs from JSON: {json_file}")
        
        with open(json_file, 'r') as f:
            jobs = json.load(f)
        
        results = {"success": [], "failed": []}
        
        for i, job_data in enumerate(jobs):
            try:
                # Validate required fields
                if not all(k in job_data for k in ['title', 'description']):
                    raise ValueError("Missing required fields: title or description")
                
                # Ensure job has an ID
                if 'id' not in job_data:
                    job_data['id'] = f"job_{datetime.now().timestamp()}_{i}"
                
                # Add to vector store
                self.vector_store.add_position(job_data)
                
                results["success"].append(job_data['id'])
                self.imported_count += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(jobs)} jobs")
                    
            except Exception as e:
                logger.error(f"Failed to import job {i}: {e}")
                results["failed"].append({"index": i, "error": str(e)})
                self.failed_count += 1
        
        return results
    
    def import_from_csv(self, csv_file: str) -> Dict[str, Any]:
        """Import jobs from CSV file"""
        logger.info(f"Importing jobs from CSV: {csv_file}")
        
        df = pd.read_csv(csv_file)
        results = {"success": [], "failed": []}
        
        # Expected CSV columns
        required_columns = ['title', 'description']
        optional_columns = [
            'department', 'location', 'work_mode', 'required_skills',
            'min_experience_years', 'max_experience_years', 'salary_min', 'salary_max'
        ]
        
        # Validate columns
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        for index, row in df.iterrows():
            try:
                job_data = {
                    "id": f"job_csv_{index}_{datetime.now().timestamp()}",
                    "title": row['title'],
                    "description": row['description'],
                    "department": row.get('department', 'General'),
                    "location": row.get('location', 'Not Specified'),
                    "work_mode": row.get('work_mode', 'onsite'),
                    "required_skills": row.get('required_skills', '').split(',') if pd.notna(row.get('required_skills')) else [],
                    "min_experience_years": float(row.get('min_experience_years', 0)),
                    "max_experience_years": float(row.get('max_experience_years', 10)) if pd.notna(row.get('max_experience_years')) else None,
                    "salary_range_min": float(row.get('salary_min', 0)) if pd.notna(row.get('salary_min')) else None,
                    "salary_range_max": float(row.get('salary_max', 0)) if pd.notna(row.get('salary_max')) else None,
                    "is_active": True
                }
                
                # Parse with job parser agent for better structure
                parse_result = self.job_parser.process({
                    "job_description": f"{job_data['title']}\n\n{job_data['description']}",
                    "position_id": job_data['id']
                })
                
                if parse_result["success"]:
                    # Use parsed data
                    parsed_position = parse_result["position"]
                    # Merge with CSV data (CSV data takes precedence for specific fields)
                    for key in ['salary_range_min', 'salary_range_max', 'location', 'department']:
                        if job_data.get(key):
                            parsed_position[key] = job_data[key]
                    
                    self.vector_store.add_position(parsed_position)
                    results["success"].append(parsed_position['id'])
                else:
                    # Fallback to direct import
                    self.vector_store.add_position(job_data)
                    results["success"].append(job_data['id'])
                
                self.imported_count += 1
                
                if (index + 1) % 10 == 0:
                    logger.info(f"Processed {index + 1}/{len(df)} jobs")
                    
            except Exception as e:
                logger.error(f"Failed to import row {index}: {e}")
                results["failed"].append({"row": index, "error": str(e)})
                self.failed_count += 1
        
        return results
    
    def import_from_directory(self, directory: str, file_pattern: str = "*.txt") -> Dict[str, Any]:
        """Import jobs from directory of files"""
        import glob
        
        logger.info(f"Importing jobs from directory: {directory}")
        
        files = glob.glob(os.path.join(directory, file_pattern))
        results = {"success": [], "failed": []}
        
        for i, file_path in enumerate(files):
            try:
                # Parse job from file
                result = self.job_parser.process({
                    "file_path": file_path,
                    "position_id": f"job_file_{i}_{os.path.basename(file_path)}"
                })
                
                if result["success"]:
                    position = result["position"]
                    self.vector_store.add_position(position)
                    results["success"].append(position['id'])
                    self.imported_count += 1
                else:
                    raise Exception(result.get("error", "Unknown error"))
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(files)} files")
                    
            except Exception as e:
                logger.error(f"Failed to import {file_path}: {e}")
                results["failed"].append({"file": file_path, "error": str(e)})
                self.failed_count += 1
        
        return results
    
    def import_from_api(self, api_url: str, headers: Dict = None) -> Dict[str, Any]:
        """Import jobs from external API"""
        import requests
        
        logger.info(f"Importing jobs from API: {api_url}")
        
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        jobs = response.json()
        if isinstance(jobs, dict) and 'data' in jobs:
            jobs = jobs['data']  # Handle wrapped responses
        
        return self.import_from_json_data(jobs)
    
    def import_from_json_data(self, jobs: List[Dict]) -> Dict[str, Any]:
        """Import jobs from JSON data (already loaded)"""
        results = {"success": [], "failed": []}
        
        for i, job_data in enumerate(jobs):
            try:
                # Map external API fields to our schema if needed
                mapped_job = self._map_external_job(job_data)
                
                # Add to vector store
                self.vector_store.add_position(mapped_job)
                results["success"].append(mapped_job['id'])
                self.imported_count += 1
                
            except Exception as e:
                logger.error(f"Failed to import job {i}: {e}")
                results["failed"].append({"index": i, "error": str(e)})
                self.failed_count += 1
        
        return results
    
    def _map_external_job(self, external_job: Dict) -> Dict:
        """Map external job format to our schema"""
        # Common field mappings
        field_mappings = {
            'job_title': 'title',
            'job_description': 'description',
            'company_name': 'company',
            'job_location': 'location',
            'salary_minimum': 'salary_range_min',
            'salary_maximum': 'salary_range_max',
            'experience_required': 'min_experience_years'
        }
        
        mapped = {}
        
        # Direct fields
        for ext_field, our_field in field_mappings.items():
            if ext_field in external_job:
                mapped[our_field] = external_job[ext_field]
        
        # Ensure required fields
        mapped['id'] = external_job.get('id', f"ext_job_{datetime.now().timestamp()}")
        mapped['title'] = mapped.get('title', external_job.get('title', 'Unknown Position'))
        mapped['description'] = mapped.get('description', external_job.get('description', ''))
        
        # Parse skills if in string format
        if 'skills' in external_job and isinstance(external_job['skills'], str):
            mapped['required_skills'] = [s.strip() for s in external_job['skills'].split(',')]
        
        return mapped
    
    def get_summary(self) -> Dict[str, Any]:
        """Get import summary"""
        return {
            "total_processed": self.imported_count + self.failed_count,
            "successfully_imported": self.imported_count,
            "failed": self.failed_count,
            "success_rate": (self.imported_count / (self.imported_count + self.failed_count) * 100) 
                          if (self.imported_count + self.failed_count) > 0 else 0
        }


def main():
    parser = argparse.ArgumentParser(description='Bulk import jobs into HR Resume Matcher')
    parser.add_argument('source', choices=['json', 'csv', 'directory', 'api', 'generate'],
                       help='Source type for import')
    parser.add_argument('path', help='Path to file/directory/API URL')
    parser.add_argument('--pattern', default='*.txt', help='File pattern for directory import')
    parser.add_argument('--api-headers', type=json.loads, help='API headers as JSON')
    parser.add_argument('--clear-existing', action='store_true', help='Clear existing jobs first')
    
    args = parser.parse_args()
    
    importer = JobBulkImporter()
    
    # Clear existing if requested
    if args.clear_existing:
        logger.info("Clearing existing jobs from vector store...")
        importer.vector_store.clear_all()
    
    # Special case: generate test data
    if args.source == 'generate' and args.path == 'test':
        logger.info("Generating 300 test job positions...")
        from tests.data.test_data_generator import JobPositionGenerator
        
        generator = JobPositionGenerator()
        positions = generator.generate_positions(300)
        
        # Save to file
        output_file = 'generated_jobs_300.json'
        with open(output_file, 'w') as f:
            json.dump(positions, f, indent=2)
        
        # Import them
        results = importer.import_from_json(output_file)
    
    # Import based on source
    elif args.source == 'json':
        results = importer.import_from_json(args.path)
    elif args.source == 'csv':
        results = importer.import_from_csv(args.path)
    elif args.source == 'directory':
        results = importer.import_from_directory(args.path, args.pattern)
    elif args.source == 'api':
        results = importer.import_from_api(args.path, args.api_headers)
    
    # Print summary
    summary = importer.get_summary()
    print("\n" + "=" * 50)
    print("IMPORT SUMMARY")
    print("=" * 50)
    print(f"Total Processed: {summary['total_processed']}")
    print(f"Successfully Imported: {summary['successfully_imported']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    # Save detailed results
    results_file = f"import_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "summary": summary,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
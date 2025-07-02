import asyncio
import unittest
import os
import sys
import json
import tempfile
import shutil
from typing import List, Dict, Any
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.orchestrator_agent import OrchestratorAgent
from agents.resume_parser_agent import ResumeParserAgent
from agents.job_parser_agent import JobParserAgent
from agents.matching_agent import MatchingAgent
from agents.salary_research_agent import SalaryResearchAgent
from agents.aspiration_agent import AspirationAgent
from tools.vector_store import VectorStoreManager
from mcp_server.tools.resume2post_tool import Resume2PostTool
from tests.data.test_data_generator import generate_test_data

class FullPipelineIntegrationTest(unittest.TestCase):
    """Integration tests for the complete HR matching pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.orchestrator = OrchestratorAgent()
        cls.test_dir = tempfile.mkdtemp()
        cls.vector_store = VectorStoreManager()
        
        # Clear vector store
        cls.vector_store.clear_all()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        cls.vector_store.clear_all()
        
    def test_end_to_end_single_match(self):
        """Test complete pipeline for single resume-position match"""
        # Create test resume file
        resume_content = """
        John Doe
        john.doe@email.com | (555) 123-4567 | San Francisco, CA
        
        SUMMARY
        Senior Software Engineer with 8 years of experience in building scalable applications.
        Expert in Python, AWS, and machine learning technologies.
        
        EXPERIENCE
        Senior Software Engineer at Tech Corp (2020 - Present)
        - Developed microservices using Python and FastAPI
        - Implemented ML models for recommendation system
        - Led team of 5 engineers
        
        EDUCATION
        BS Computer Science - UC Berkeley (2016)
        
        SKILLS
        Python, FastAPI, Django, AWS, Docker, Kubernetes, PostgreSQL, Machine Learning
        """
        
        resume_path = os.path.join(self.test_dir, "test_resume.txt")
        with open(resume_path, 'w') as f:
            f.write(resume_content)
        
        # Create test job file
        job_content = """
        Senior Software Engineer - ML Platform
        
        Location: San Francisco, CA
        Department: Engineering
        
        We are looking for a Senior Software Engineer to join our ML Platform team.
        
        Requirements:
        - 5+ years of software engineering experience
        - Strong Python skills
        - Experience with AWS and containerization
        - Machine learning experience preferred
        
        Responsibilities:
        - Build scalable ML infrastructure
        - Develop APIs for model serving
        - Mentor junior engineers
        
        Salary: $140,000 - $180,000
        """
        
        job_path = os.path.join(self.test_dir, "test_job.txt")
        with open(job_path, 'w') as f:
            f.write(job_content)
        
        # Run end-to-end match
        result = self.orchestrator.process_single_match(
            resume_path=resume_path,
            position_path=job_path,
            include_salary=True,
            include_aspirations=True
        )
        
        # Validate results
        self.assertTrue(result["success"], "Match should succeed")
        self.assertIn("match", result)
        self.assertIn("resume", result)
        self.assertIn("position", result)
        
        # Check match quality
        match = result["match"]
        self.assertGreater(match["overall_score"], 0.7, "Should be a good match")
        self.assertGreater(match["skill_match_score"], 0.6, "Skills should match well")
        
        # Check salary research
        if "salary_research" in result:
            salary_data = result["salary_research"]["salary_research"]
            self.assertIn("market_average", salary_data)
            self.assertGreater(salary_data["market_average"], 0)
        
        # Check aspiration analysis
        if "aspiration_analysis" in result:
            aspirations = result["aspiration_analysis"]
            self.assertIn("career_trajectory", aspirations)
            self.assertIn("insights", aspirations)
    
    def test_batch_processing_pipeline(self):
        """Test batch processing with multiple resumes and positions"""
        # Generate test data
        from tests.data.test_data_generator import ResumeGenerator, JobPositionGenerator
        
        resume_gen = ResumeGenerator()
        job_gen = JobPositionGenerator()
        
        # Generate 5 resumes and 10 positions
        resumes = resume_gen.generate_resumes(5)
        positions = job_gen.generate_positions(10)
        
        # Save to files
        resume_files = []
        for i, resume in enumerate(resumes):
            path = os.path.join(self.test_dir, f"resume_{i}.json")
            with open(path, 'w') as f:
                f.write(resume["raw_text"])
            resume_files.append(path)
        
        position_files = []
        for i, position in enumerate(positions):
            path = os.path.join(self.test_dir, f"position_{i}.json") 
            with open(path, 'w') as f:
                f.write(position["description"])
            position_files.append(path)
        
        # Process first resume against all positions
        start_time = time.time()
        
        matches = []
        for pos_file in position_files[:5]:  # Test with 5 positions
            result = self.orchestrator.process_single_match(
                resume_path=resume_files[0],
                position_path=pos_file,
                include_salary=False,  # Skip for speed
                include_aspirations=False
            )
            
            if result["success"]:
                matches.append({
                    "position": result["position"]["title"],
                    "score": result["match"]["overall_score"]
                })
        
        processing_time = time.time() - start_time
        
        # Validate batch results
        self.assertGreater(len(matches), 0, "Should have some matches")
        self.assertLess(processing_time, 30, "Batch should complete within 30 seconds")
        
        # Check ranking
        matches.sort(key=lambda x: x["score"], reverse=True)
        for i in range(1, len(matches)):
            self.assertLessEqual(matches[i]["score"], matches[i-1]["score"],
                               "Matches should be properly ranked")
    
    def test_agent_coordination(self):
        """Test that all agents work together correctly"""
        # Create minimal test data
        resume_data = {
            "id": "test_resume",
            "name": "Test User",
            "email": "test@email.com",
            "skills": [{"name": "Python"}, {"name": "AWS"}],
            "total_experience_years": 5,
            "raw_text": "Test resume content"
        }
        
        position_data = {
            "id": "test_position",
            "title": "Software Engineer",
            "required_skills": ["Python", "AWS"],
            "min_experience_years": 3,
            "location": "Remote",
            "salary_range_min": 100000,
            "salary_range_max": 150000
        }
        
        # Test individual agents
        agents_tests = [
            ("Matching Agent", self.orchestrator.matching_agent, {
                "resume": resume_data,
                "position": position_data
            }),
            ("Salary Research Agent", self.orchestrator.salary_agent, {
                "position_title": position_data["title"],
                "location": position_data["location"],
                "experience_years": resume_data["total_experience_years"]
            }),
            ("Aspiration Agent", self.orchestrator.aspiration_agent, {
                "resume_data": resume_data,
                "position_data": position_data
            })
        ]
        
        for agent_name, agent, test_data in agents_tests:
            with self.subTest(agent=agent_name):
                result = agent.process(test_data)
                self.assertTrue(result["success"], 
                              f"{agent_name} should process successfully")
    
    def test_vector_store_integration(self):
        """Test vector store integration with matching"""
        # Add test data to vector store
        test_positions = []
        for i in range(5):
            position = {
                "id": f"pos_{i}",
                "title": f"Position {i}",
                "description": f"Test position {i}",
                "required_skills": ["Python", "AWS"] if i < 3 else ["Java", "Spring"],
                "location": "San Francisco",
                "min_experience_years": 3
            }
            self.vector_store.add_position(position)
            test_positions.append(position)
        
        test_resume = {
            "id": "test_resume_vs",
            "name": "Vector Test",
            "skills": [{"name": "Python"}, {"name": "AWS"}],
            "total_experience_years": 5,
            "raw_text": "Python developer with AWS experience"
        }
        
        # Search for similar positions
        results = self.vector_store.search_similar_positions(
            resume_text=test_resume["raw_text"],
            k=3
        )
        
        # Validate results
        self.assertEqual(len(results), 3, "Should return 3 results")
        
        # Python/AWS positions should rank higher
        top_result = results[0]
        position = next(p for p in test_positions if p["id"] == top_result["position_id"])
        self.assertIn("Python", position["required_skills"],
                     "Top match should require Python")
    
    def test_mcp_integration(self):
        """Test MCP server integration"""
        mcp_tool = Resume2PostTool()
        
        # Create test resume file
        resume_path = os.path.join(self.test_dir, "mcp_test_resume.txt")
        with open(resume_path, 'w') as f:
            f.write("Python developer with 5 years experience")
        
        # Test categorization
        async def run_categorization():
            result = await mcp_tool.categorize_resume_to_posts(
                resume_path=resume_path,
                top_k=5,
                min_confidence="low"
            )
            return result
        
        # Run async test
        result = asyncio.run(run_categorization())
        
        # Validate MCP results
        self.assertIsNotNone(result, "Should return categorization result")
        self.assertGreater(result.processing_time, 0, "Should track processing time")
    
    def test_error_handling(self):
        """Test error handling across the pipeline"""
        # Test with non-existent file
        result = self.orchestrator.process_single_match(
            resume_path="/non/existent/file.txt",
            position_path="/another/missing/file.txt"
        )
        
        self.assertFalse(result["success"], "Should fail gracefully")
        self.assertIn("error", result)
        
        # Test with invalid data
        invalid_resume = {"invalid": "data"}
        invalid_position = {"no_required_fields": True}
        
        match_result = self.orchestrator.matching_agent.process({
            "resume": invalid_resume,
            "position": invalid_position
        })
        
        # Should handle gracefully even with invalid data
        self.assertIn("success", match_result)


class PerformanceIntegrationTest(unittest.TestCase):
    """Integration tests focused on performance"""
    
    def setUp(self):
        self.orchestrator = OrchestratorAgent()
        
    def test_concurrent_request_handling(self):
        """Test handling multiple concurrent requests"""
        import concurrent.futures
        
        # Create test data
        resume_data = {
            "id": "perf_test",
            "name": "Performance Test",
            "skills": [{"name": "Python"}],
            "total_experience_years": 5,
            "raw_text": "Test resume"
        }
        
        position_data = {
            "id": "perf_pos",
            "title": "Test Position",
            "required_skills": ["Python"],
            "min_experience_years": 3
        }
        
        # Function to run single match
        def run_match():
            return self.orchestrator.matching_agent.process({
                "resume": resume_data,
                "position": position_data
            })
        
        # Run concurrent matches
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_match) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Validate results
        self.assertEqual(len(results), 10, "All requests should complete")
        self.assertTrue(all(r["success"] for r in results), "All should succeed")
        self.assertLess(total_time, 10, "Concurrent processing should be fast")
        
        # Check consistency
        scores = [r["match_result"]["overall_score"] for r in results]
        self.assertEqual(len(set(scores)), 1, "Scores should be consistent")
    
    def test_memory_efficiency(self):
        """Test memory usage with large datasets"""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Get initial memory
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple matches
        for i in range(20):
            resume = {
                "id": f"mem_test_{i}",
                "name": f"Memory Test {i}",
                "skills": [{"name": "Python"}] * 50,  # Large skill list
                "total_experience_years": 5,
                "raw_text": "Large resume text " * 1000  # Large text
            }
            
            position = {
                "id": f"mem_pos_{i}",
                "title": "Memory Test Position",
                "required_skills": ["Python"] * 20,
                "description": "Large description " * 500
            }
            
            self.orchestrator.matching_agent.process({
                "resume": resume,
                "position": position
            })
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        self.assertLess(memory_increase, 500, 
                       f"Memory increase ({memory_increase}MB) should be reasonable")


def run_integration_tests():
    """Run all integration tests with reporting"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(FullPipelineIntegrationTest))
    suite.addTest(unittest.makeSuite(PerformanceIntegrationTest))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    report = {
        "test_date": datetime.now().isoformat(),
        "test_type": "integration",
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) 
                        / result.testsRun * 100) if result.testsRun > 0 else 0,
        "test_details": {
            "pipeline_tests": "FullPipelineIntegrationTest",
            "performance_tests": "PerformanceIntegrationTest"
        }
    }
    
    # Save report
    report_path = "tests/reports/integration_test_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Integration test report saved to: {report_path}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
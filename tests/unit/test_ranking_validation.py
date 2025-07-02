import unittest
import asyncio
import json
import os
import sys
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.matching_agent import MatchingAgent
from mcp_server.models.resume_categorizer import Resume2PostCategorizer
from tests.data.test_data_generator import JobPositionGenerator, ResumeGenerator

class RankingValidationTest(unittest.TestCase):
    """Test suite for validating ranking accuracy"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.matching_agent = MatchingAgent()
        cls.categorizer = Resume2PostCategorizer()
        cls.job_generator = JobPositionGenerator()
        cls.resume_generator = ResumeGenerator()
        
    def test_skill_based_ranking(self):
        """Test that candidates with more matching skills rank higher"""
        # Create a job with specific skills
        job = {
            "id": "test_job_1",
            "title": "Software Engineer",
            "required_skills": ["Python", "Django", "PostgreSQL", "Docker", "AWS"],
            "preferred_skills": ["React", "Redis", "Kubernetes"],
            "min_experience_years": 3,
            "max_experience_years": 8,
            "education_requirements": ["bachelors"]
        }
        
        # Create resumes with varying skill matches
        resumes = [
            # Resume 1: Perfect skill match
            {
                "id": "resume_1",
                "name": "Perfect Match",
                "skills": [
                    {"name": "Python"}, {"name": "Django"}, {"name": "PostgreSQL"},
                    {"name": "Docker"}, {"name": "AWS"}, {"name": "React"}, {"name": "Redis"}
                ],
                "total_experience_years": 5,
                "education": [{"level": "bachelors"}]
            },
            # Resume 2: Good skill match
            {
                "id": "resume_2", 
                "name": "Good Match",
                "skills": [
                    {"name": "Python"}, {"name": "Django"}, {"name": "PostgreSQL"},
                    {"name": "Docker"}
                ],
                "total_experience_years": 4,
                "education": [{"level": "bachelors"}]
            },
            # Resume 3: Partial skill match
            {
                "id": "resume_3",
                "name": "Partial Match",
                "skills": [
                    {"name": "Python"}, {"name": "Django"}
                ],
                "total_experience_years": 3,
                "education": [{"level": "bachelors"}]
            },
            # Resume 4: Poor skill match
            {
                "id": "resume_4",
                "name": "Poor Match",
                "skills": [
                    {"name": "Java"}, {"name": "Spring"}
                ],
                "total_experience_years": 5,
                "education": [{"level": "bachelors"}]
            }
        ]
        
        # Match all resumes against the job
        matches = []
        for resume in resumes:
            result = self.matching_agent.process({
                "resume": resume,
                "position": job
            })
            
            self.assertTrue(result["success"])
            match_data = result["match_result"]
            matches.append({
                "resume_id": resume["id"],
                "name": resume["name"],
                "overall_score": match_data["overall_score"],
                "skill_score": match_data["skill_match_score"]
            })
        
        # Sort by overall score
        matches.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Validate ranking
        self.assertEqual(matches[0]["resume_id"], "resume_1", "Perfect match should rank first")
        self.assertEqual(matches[1]["resume_id"], "resume_2", "Good match should rank second")
        self.assertEqual(matches[2]["resume_id"], "resume_3", "Partial match should rank third")
        self.assertEqual(matches[3]["resume_id"], "resume_4", "Poor match should rank last")
        
        # Validate score differences
        self.assertGreater(matches[0]["skill_score"], 0.8, "Perfect match should have high skill score")
        self.assertLess(matches[3]["skill_score"], 0.3, "Poor match should have low skill score")
        
    def test_experience_based_ranking(self):
        """Test that experience affects ranking appropriately"""
        job = {
            "id": "test_job_2",
            "title": "Senior Software Engineer",
            "required_skills": ["Python"],
            "min_experience_years": 5,
            "max_experience_years": 10,
            "experience_level": "senior"
        }
        
        resumes = [
            # Optimal experience
            {
                "id": "resume_opt",
                "name": "Optimal Experience",
                "skills": [{"name": "Python"}],
                "total_experience_years": 7
            },
            # Under-qualified
            {
                "id": "resume_under",
                "name": "Under-qualified",
                "skills": [{"name": "Python"}],
                "total_experience_years": 2
            },
            # Over-qualified
            {
                "id": "resume_over",
                "name": "Over-qualified",
                "skills": [{"name": "Python"}],
                "total_experience_years": 15
            }
        ]
        
        matches = []
        for resume in resumes:
            result = self.matching_agent.process({
                "resume": resume,
                "position": job
            })
            match_data = result["match_result"]
            matches.append({
                "resume_id": resume["id"],
                "experience_score": match_data["experience_match_score"],
                "overall_score": match_data["overall_score"]
            })
        
        # Find scores by resume ID
        opt_score = next(m for m in matches if m["resume_id"] == "resume_opt")
        under_score = next(m for m in matches if m["resume_id"] == "resume_under")
        over_score = next(m for m in matches if m["resume_id"] == "resume_over")
        
        # Validate experience scoring
        self.assertEqual(opt_score["experience_score"], 1.0, "Optimal experience should score 1.0")
        self.assertLess(under_score["experience_score"], 0.7, "Under-qualified should score lower")
        self.assertLess(over_score["experience_score"], 1.0, "Over-qualified should not score perfect")
        self.assertGreater(over_score["experience_score"], under_score["experience_score"], 
                          "Over-qualified should score better than under-qualified")
        
    def test_comprehensive_ranking(self):
        """Test ranking with all factors combined"""
        job = {
            "id": "test_job_3",
            "title": "Full Stack Engineer",
            "department": "Engineering",
            "location": "San Francisco, CA",
            "work_mode": "hybrid",
            "required_skills": ["Python", "React", "PostgreSQL", "AWS", "Docker"],
            "preferred_skills": ["GraphQL", "TypeScript", "Redis"],
            "min_experience_years": 4,
            "max_experience_years": 8,
            "education_requirements": ["bachelors"],
            "salary_range_min": 120000,
            "salary_range_max": 180000,
            "description": "Looking for a full stack engineer to build scalable web applications"
        }
        
        # Generate diverse resumes
        test_resumes = []
        
        # Resume 1: Ideal candidate
        test_resumes.append({
            "id": "ideal_candidate",
            "name": "Ideal Candidate",
            "skills": [
                {"name": "Python"}, {"name": "React"}, {"name": "PostgreSQL"},
                {"name": "AWS"}, {"name": "Docker"}, {"name": "GraphQL"},
                {"name": "TypeScript"}
            ],
            "total_experience_years": 6,
            "education": [{"level": "bachelors", "field": "Computer Science"}],
            "location": "San Francisco, CA",
            "expected_salary": 150000,
            "raw_text": "Experienced full stack engineer with Python and React expertise..."
        })
        
        # Resume 2: Strong technical, wrong location
        test_resumes.append({
            "id": "wrong_location",
            "name": "Wrong Location",
            "skills": [
                {"name": "Python"}, {"name": "React"}, {"name": "PostgreSQL"},
                {"name": "AWS"}, {"name": "Docker"}
            ],
            "total_experience_years": 5,
            "education": [{"level": "bachelors"}],
            "location": "New York, NY",
            "expected_salary": 140000,
            "raw_text": "Full stack developer based in NYC..."
        })
        
        # Resume 3: Junior but high potential
        test_resumes.append({
            "id": "high_potential",
            "name": "High Potential Junior",
            "skills": [
                {"name": "Python"}, {"name": "React"}, {"name": "PostgreSQL"}
            ],
            "total_experience_years": 2,
            "education": [{"level": "masters", "field": "Computer Science"}],
            "location": "San Francisco, CA",
            "expected_salary": 110000,
            "raw_text": "Recent MS grad with strong fundamentals..."
        })
        
        # Resume 4: Overqualified expensive
        test_resumes.append({
            "id": "overqualified",
            "name": "Overqualified Senior",
            "skills": [
                {"name": "Python"}, {"name": "React"}, {"name": "PostgreSQL"},
                {"name": "AWS"}, {"name": "Docker"}, {"name": "Kubernetes"},
                {"name": "Go"}, {"name": "Rust"}
            ],
            "total_experience_years": 12,
            "education": [{"level": "masters"}],
            "location": "San Francisco, CA",
            "expected_salary": 220000,
            "raw_text": "Senior architect with 12 years experience..."
        })
        
        # Match and rank
        detailed_matches = []
        for resume in test_resumes:
            result = self.matching_agent.process({
                "resume": resume,
                "position": job
            })
            
            if result["success"]:
                match_data = result["match_result"]
                detailed_matches.append({
                    "resume_id": resume["id"],
                    "name": resume["name"],
                    "overall_score": match_data["overall_score"],
                    "skill_score": match_data["skill_match_score"],
                    "experience_score": match_data["experience_match_score"],
                    "education_score": match_data["education_match_score"],
                    "salary_score": match_data["salary_compatibility_score"],
                    "strengths": match_data["strengths"][:2],
                    "gaps": match_data["gaps"][:2]
                })
        
        # Sort by overall score
        detailed_matches.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Print ranking for analysis
        print("\nComprehensive Ranking Results:")
        print("=" * 80)
        for i, match in enumerate(detailed_matches):
            print(f"\nRank {i+1}: {match['name']} (ID: {match['resume_id']})")
            print(f"  Overall Score: {match['overall_score']:.2%}")
            print(f"  Components: Skill={match['skill_score']:.2f}, "
                  f"Exp={match['experience_score']:.2f}, "
                  f"Edu={match['education_score']:.2f}, "
                  f"Salary={match['salary_score']:.2f}")
            print(f"  Strengths: {', '.join(match['strengths'])}")
            print(f"  Gaps: {', '.join(match['gaps'])}")
        
        # Validate expected ranking logic
        ideal_match = next(m for m in detailed_matches if m["resume_id"] == "ideal_candidate")
        self.assertGreater(ideal_match["overall_score"], 0.75, 
                          "Ideal candidate should have high overall score")
        
        # Overqualified should not rank first due to salary mismatch
        overqualified_rank = next(i for i, m in enumerate(detailed_matches) 
                                 if m["resume_id"] == "overqualified")
        self.assertGreater(overqualified_rank, 0, 
                          "Overqualified candidate should not rank first due to salary")
        
    def test_ranking_stability(self):
        """Test that ranking is stable and consistent"""
        # Create test data
        job = self.job_generator.generate_job_position(1)
        resumes = [self.resume_generator.generate_resume() for _ in range(5)]
        
        # Run matching multiple times
        rankings = []
        for _ in range(3):
            matches = []
            for resume in resumes:
                result = self.matching_agent.process({
                    "resume": resume,
                    "position": job
                })
                if result["success"]:
                    matches.append({
                        "resume_id": resume["id"],
                        "score": result["match_result"]["overall_score"]
                    })
            
            # Sort and get ranking
            matches.sort(key=lambda x: x["score"], reverse=True)
            ranking = [m["resume_id"] for m in matches]
            rankings.append(ranking)
        
        # Verify rankings are consistent
        for i in range(1, len(rankings)):
            self.assertEqual(rankings[0], rankings[i], 
                           f"Ranking should be consistent across runs")
        
    def test_score_distribution(self):
        """Test that scores are well distributed"""
        # Generate varied resumes and a job
        job = self.job_generator.generate_job_position(1)
        resumes = []
        
        # Create resumes with different match levels
        for i in range(20):
            if i < 5:  # Good matches
                profile = "senior"
            elif i < 10:  # Medium matches
                profile = "mid" 
            else:  # Poor matches
                profile = "junior"
            
            resume = self.resume_generator.generate_resume(profile)
            resumes.append(resume)
        
        # Match all resumes
        scores = []
        for resume in resumes:
            result = self.matching_agent.process({
                "resume": resume,
                "position": job
            })
            if result["success"]:
                scores.append(result["match_result"]["overall_score"])
        
        # Analyze score distribution
        scores.sort(reverse=True)
        
        # Check for good distribution
        self.assertGreater(max(scores), 0.6, "Best matches should score > 60%")
        self.assertLess(min(scores), 0.4, "Poor matches should score < 40%")
        
        # Check for gradual decrease
        for i in range(1, len(scores)):
            self.assertLessEqual(scores[i], scores[i-1], 
                               "Scores should be in descending order")
        
        # Calculate score variance
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        self.assertGreater(std_score, 0.1, 
                          "Scores should have meaningful variance")
        
        print(f"\nScore Distribution Stats:")
        print(f"  Mean: {mean_score:.3f}")
        print(f"  Std Dev: {std_score:.3f}")
        print(f"  Range: {min(scores):.3f} - {max(scores):.3f}")
    
    def test_mcp_categorizer_ranking(self):
        """Test MCP categorizer ranking accuracy"""
        # Generate test data
        positions = [self.job_generator.generate_job_position(i) for i in range(10)]
        resume = self.resume_generator.generate_resume("senior")
        
        # Save resume temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(resume, f)
            resume_path = f.name
        
        try:
            # Categorize resume
            results = self.categorizer.categorize_resume(resume_path, top_k=5)
            
            # Verify results are ranked by score
            for i in range(1, len(results)):
                self.assertGreaterEqual(results[i-1]["final_score"], 
                                      results[i]["final_score"],
                                      "Results should be sorted by score")
            
            # Verify score components make sense
            for result in results:
                self.assertGreaterEqual(result["semantic_score"], 0)
                self.assertLessEqual(result["semantic_score"], 1)
                self.assertGreaterEqual(result["feature_score"], 0) 
                self.assertLessEqual(result["feature_score"], 1)
                
                # Final score should be weighted average
                expected_final = (self.categorizer.semantic_weight * result["semantic_score"] +
                                self.categorizer.feature_weight * result["feature_score"])
                self.assertAlmostEqual(result["final_score"], expected_final, places=3)
                
        finally:
            # Cleanup
            os.unlink(resume_path)


class RankingEdgeCaseTest(unittest.TestCase):
    """Test edge cases in ranking"""
    
    def setUp(self):
        self.matching_agent = MatchingAgent()
        
    def test_empty_skills_ranking(self):
        """Test ranking when resume has no skills"""
        job = {
            "id": "job_1",
            "required_skills": ["Python", "Django"],
            "min_experience_years": 3
        }
        
        resume = {
            "id": "resume_1",
            "name": "No Skills",
            "skills": [],
            "total_experience_years": 5
        }
        
        result = self.matching_agent.process({
            "resume": resume,
            "position": job
        })
        
        self.assertTrue(result["success"])
        self.assertEqual(result["match_result"]["skill_match_score"], 0,
                        "No skills should result in 0 skill score")
        
    def test_identical_candidates_ranking(self):
        """Test ranking when candidates are identical"""
        job = {
            "id": "job_1",
            "required_skills": ["Python"],
            "min_experience_years": 3
        }
        
        # Create two identical resumes
        resume_base = {
            "skills": [{"name": "Python"}],
            "total_experience_years": 5,
            "education": [{"level": "bachelors"}]
        }
        
        resume1 = {**resume_base, "id": "resume_1", "name": "Candidate 1"}
        resume2 = {**resume_base, "id": "resume_2", "name": "Candidate 2"}
        
        result1 = self.matching_agent.process({"resume": resume1, "position": job})
        result2 = self.matching_agent.process({"resume": resume2, "position": job})
        
        score1 = result1["match_result"]["overall_score"]
        score2 = result2["match_result"]["overall_score"]
        
        self.assertAlmostEqual(score1, score2, places=3,
                             "Identical candidates should have identical scores")


def run_ranking_tests():
    """Run all ranking validation tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTest(unittest.makeSuite(RankingValidationTest))
    suite.addTest(unittest.makeSuite(RankingEdgeCaseTest))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    report = {
        "test_date": datetime.now().isoformat(),
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) 
                        / result.testsRun * 100) if result.testsRun > 0 else 0
    }
    
    # Save report
    report_path = "tests/reports/ranking_validation_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Ranking validation report saved to: {report_path}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_ranking_tests()
    sys.exit(0 if success else 1)
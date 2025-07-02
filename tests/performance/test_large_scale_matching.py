import asyncio
import time
import json
import os
import sys
from typing import List, Dict, Any, Tuple
from datetime import datetime
import statistics
import psutil
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.orchestrator_agent import OrchestratorAgent
from agents.resume_parser_agent import ResumeParserAgent
from agents.job_parser_agent import JobParserAgent
from agents.matching_agent import MatchingAgent
from tools.vector_store import VectorStoreManager
from models.schemas import BatchMatchRequest
from tests.data.test_data_generator import generate_test_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Track and calculate performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_start = None
        self.memory_peak = None
        self.processing_times = []
        self.match_scores = []
        self.errors = []
        
    def start(self):
        """Start tracking metrics"""
        self.start_time = time.time()
        self.memory_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        tracemalloc.start()
        
    def end(self):
        """End tracking and calculate final metrics"""
        self.end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        self.memory_peak = peak / 1024 / 1024  # MB
        tracemalloc.stop()
        
    def add_processing_time(self, time_seconds: float):
        """Add individual processing time"""
        self.processing_times.append(time_seconds)
        
    def add_match_score(self, score: float):
        """Add match score"""
        self.match_scores.append(score)
        
    def add_error(self, error: str):
        """Add error"""
        self.errors.append(error)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_time = self.end_time - self.start_time if self.end_time else 0
        
        return {
            "total_time_seconds": round(total_time, 2),
            "average_processing_time": round(statistics.mean(self.processing_times), 3) if self.processing_times else 0,
            "median_processing_time": round(statistics.median(self.processing_times), 3) if self.processing_times else 0,
            "min_processing_time": round(min(self.processing_times), 3) if self.processing_times else 0,
            "max_processing_time": round(max(self.processing_times), 3) if self.processing_times else 0,
            "memory_used_mb": round(self.memory_peak - self.memory_start, 2) if self.memory_peak else 0,
            "memory_peak_mb": round(self.memory_peak, 2) if self.memory_peak else 0,
            "total_matches": len(self.match_scores),
            "average_match_score": round(statistics.mean(self.match_scores), 3) if self.match_scores else 0,
            "error_count": len(self.errors),
            "success_rate": round((1 - len(self.errors) / max(len(self.processing_times), 1)) * 100, 2)
        }


class LargeScaleMatchingTest:
    """Test large-scale matching with 300 positions"""
    
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.vector_store = VectorStoreManager()
        self.metrics = PerformanceMetrics()
        
    async def setup_test_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load or generate test data"""
        logger.info("Setting up test data...")
        
        # Check if test data exists
        positions_file = "tests/data/job_positions_300.json"
        resumes_file = "tests/data/resumes_with_perfect_matches.json"
        
        if not os.path.exists(positions_file) or not os.path.exists(resumes_file):
            logger.info("Generating test data...")
            generate_test_data()
        
        # Load data
        with open(positions_file, 'r') as f:
            positions = json.load(f)
        
        with open(resumes_file, 'r') as f:
            resumes = json.load(f)
        
        logger.info(f"Loaded {len(positions)} positions and {len(resumes)} resumes")
        return positions, resumes
    
    async def populate_vector_store(self, positions: List[Dict], resumes: List[Dict]):
        """Populate vector store with test data"""
        logger.info("Populating vector store...")
        
        # Clear existing data
        self.vector_store.clear_all()
        
        # Add positions
        start_time = time.time()
        for i, position in enumerate(positions):
            try:
                self.vector_store.add_position(position)
                if (i + 1) % 50 == 0:
                    logger.info(f"Added {i + 1} positions to vector store")
            except Exception as e:
                logger.error(f"Error adding position {position['id']}: {e}")
        
        positions_time = time.time() - start_time
        
        # Add resumes
        start_time = time.time()
        for i, resume in enumerate(resumes):
            try:
                self.vector_store.add_resume(resume)
                if (i + 1) % 10 == 0:
                    logger.info(f"Added {i + 1} resumes to vector store")
            except Exception as e:
                logger.error(f"Error adding resume {resume['id']}: {e}")
        
        resumes_time = time.time() - start_time
        
        logger.info(f"Vector store populated - Positions: {positions_time:.2f}s, Resumes: {resumes_time:.2f}s")
    
    async def test_single_resume_all_positions(self, resume: Dict, positions: List[Dict]) -> Dict[str, Any]:
        """Test matching a single resume against all 300 positions"""
        logger.info(f"Testing resume {resume['name']} against {len(positions)} positions")
        
        start_time = time.time()
        
        # Use vector similarity for initial filtering
        resume_text = resume['raw_text']
        similar_positions = self.vector_store.search_similar_positions(
            resume_text=resume_text,
            k=100  # Get top 100 matches
        )
        
        vector_search_time = time.time() - start_time
        
        # Deep match analysis for top positions
        detailed_matches = []
        matching_start = time.time()
        
        for pos_match in similar_positions[:20]:  # Detailed analysis for top 20
            try:
                # Find full position data
                position = next((p for p in positions if p['id'] == pos_match['position_id']), None)
                if position:
                    match_result = self.orchestrator.matching_agent.process({
                        "resume": resume,
                        "position": position
                    })
                    
                    if match_result["success"]:
                        match_data = match_result["match_result"]
                        detailed_matches.append({
                            "position_id": position['id'],
                            "position_title": position['title'],
                            "overall_score": match_data['overall_score'],
                            "skill_match": match_data['skill_match_score'],
                            "experience_match": match_data['experience_match_score'],
                            "vector_similarity": pos_match['similarity_score']
                        })
                        self.metrics.add_match_score(match_data['overall_score'])
            except Exception as e:
                logger.error(f"Error in detailed matching: {e}")
                self.metrics.add_error(str(e))
        
        matching_time = time.time() - matching_start
        total_time = time.time() - start_time
        
        # Sort by overall score
        detailed_matches.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return {
            "resume_id": resume['id'],
            "resume_name": resume['name'],
            "total_positions": len(positions),
            "vector_search_results": len(similar_positions),
            "detailed_matches": len(detailed_matches),
            "top_5_matches": detailed_matches[:5],
            "vector_search_time": round(vector_search_time, 3),
            "matching_time": round(matching_time, 3),
            "total_time": round(total_time, 3)
        }
    
    async def test_batch_matching(self, resumes: List[Dict], positions: List[Dict], 
                                 batch_size: int = 10) -> List[Dict[str, Any]]:
        """Test batch matching with multiple resumes"""
        logger.info(f"Testing batch matching with {len(resumes)} resumes and {len(positions)} positions")
        
        results = []
        
        # Process in batches
        for i in range(0, len(resumes), batch_size):
            batch_resumes = resumes[i:i + batch_size]
            batch_start = time.time()
            
            # Use thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                
                for resume in batch_resumes:
                    future = executor.submit(
                        asyncio.run,
                        self.test_single_resume_all_positions(resume, positions[:50])  # Test with subset
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        self.metrics.add_processing_time(result['total_time'])
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")
                        self.metrics.add_error(str(e))
            
            batch_time = time.time() - batch_start
            logger.info(f"Batch {i//batch_size + 1} completed in {batch_time:.2f}s")
        
        return results
    
    async def test_concurrent_operations(self, resumes: List[Dict], positions: List[Dict]):
        """Test system under concurrent load"""
        logger.info("Testing concurrent operations...")
        
        # Simulate multiple concurrent users
        concurrent_tasks = []
        
        for i in range(5):  # 5 concurrent operations
            resume = resumes[i % len(resumes)]
            task = self.test_single_resume_all_positions(resume, positions[:100])
            concurrent_tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        concurrent_time = time.time() - start_time
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        
        return {
            "total_concurrent_tasks": len(concurrent_tasks),
            "successful_tasks": successful,
            "total_time": round(concurrent_time, 2),
            "average_time_per_task": round(concurrent_time / len(concurrent_tasks), 2)
        }
    
    async def validate_ranking_accuracy(self, positions: List[Dict], resumes: List[Dict]):
        """Validate that ranking is accurate"""
        logger.info("Validating ranking accuracy...")
        
        # Load perfect matches
        with open("tests/data/perfect_matches.json", 'r') as f:
            perfect_matches = json.load(f)
        
        validation_results = []
        
        for perfect_match in perfect_matches:
            resume = next((r for r in resumes if r['id'] == perfect_match['resume_id']), None)
            if not resume:
                continue
            
            # Get matches for this resume
            result = await self.test_single_resume_all_positions(resume, positions)
            
            # Check if the expected position is in top 5
            expected_position_id = perfect_match['position_id']
            top_5_ids = [m['position_id'] for m in result['top_5_matches']]
            
            is_in_top_5 = expected_position_id in top_5_ids
            rank = top_5_ids.index(expected_position_id) + 1 if is_in_top_5 else -1
            
            validation_results.append({
                "resume_id": resume['id'],
                "expected_position_id": expected_position_id,
                "expected_score": perfect_match['expected_score'],
                "is_in_top_5": is_in_top_5,
                "rank": rank,
                "actual_score": next((m['overall_score'] for m in result['top_5_matches'] 
                                    if m['position_id'] == expected_position_id), 0)
            })
        
        # Calculate accuracy metrics
        accuracy = sum(1 for v in validation_results if v['is_in_top_5']) / len(validation_results) * 100
        average_rank = statistics.mean([v['rank'] for v in validation_results if v['rank'] > 0])
        
        return {
            "total_validations": len(validation_results),
            "accuracy_percentage": round(accuracy, 2),
            "average_rank": round(average_rank, 2),
            "validation_details": validation_results
        }
    
    async def run_all_tests(self):
        """Run all performance tests"""
        logger.info("=" * 50)
        logger.info("Starting Large Scale Matching Tests")
        logger.info("=" * 50)
        
        # Start metrics tracking
        self.metrics.start()
        
        # Setup test data
        positions, resumes = await self.setup_test_data()
        
        # Populate vector store
        await self.populate_vector_store(positions, resumes)
        
        # Test 1: Single resume against all positions
        logger.info("\n🧪 Test 1: Single Resume vs 300 Positions")
        single_result = await self.test_single_resume_all_positions(resumes[0], positions)
        
        # Test 2: Batch matching
        logger.info("\n🧪 Test 2: Batch Matching (10 resumes)")
        batch_results = await self.test_batch_matching(resumes[:10], positions)
        
        # Test 3: Concurrent operations
        logger.info("\n🧪 Test 3: Concurrent Operations")
        concurrent_result = await self.test_concurrent_operations(resumes, positions)
        
        # Test 4: Ranking validation
        logger.info("\n🧪 Test 4: Ranking Accuracy Validation")
        ranking_validation = await self.validate_ranking_accuracy(positions, resumes)
        
        # End metrics tracking
        self.metrics.end()
        
        # Generate report
        report = {
            "test_timestamp": datetime.now().isoformat(),
            "test_configuration": {
                "total_positions": len(positions),
                "total_resumes": len(resumes),
                "vector_search_k": 100,
                "detailed_analysis_k": 20
            },
            "performance_metrics": self.metrics.get_summary(),
            "test_results": {
                "single_resume_test": {
                    "resume_name": single_result['resume_name'],
                    "vector_search_time": single_result['vector_search_time'],
                    "matching_time": single_result['matching_time'],
                    "total_time": single_result['total_time'],
                    "top_match": single_result['top_5_matches'][0] if single_result['top_5_matches'] else None
                },
                "batch_test": {
                    "total_resumes": len(batch_results),
                    "average_time": statistics.mean([r['total_time'] for r in batch_results]),
                    "total_matches": sum([r['detailed_matches'] for r in batch_results])
                },
                "concurrent_test": concurrent_result,
                "ranking_validation": ranking_validation
            }
        }
        
        # Save report
        report_path = f"tests/reports/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_summary(report)
        
        logger.info(f"\n✅ Full report saved to: {report_path}")
        
        return report
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        
        metrics = report['performance_metrics']
        print(f"\n📊 Overall Performance:")
        print(f"   Total Time: {metrics['total_time_seconds']}s")
        print(f"   Memory Used: {metrics['memory_used_mb']} MB")
        print(f"   Success Rate: {metrics['success_rate']}%")
        
        single_test = report['test_results']['single_resume_test']
        print(f"\n🎯 Single Resume Test (300 positions):")
        print(f"   Vector Search: {single_test['vector_search_time']}s")
        print(f"   Detailed Matching: {single_test['matching_time']}s")
        print(f"   Total Time: {single_test['total_time']}s")
        
        if single_test['top_match']:
            print(f"   Top Match: {single_test['top_match']['position_title']} "
                  f"(Score: {single_test['top_match']['overall_score']:.2%})")
        
        batch_test = report['test_results']['batch_test']
        print(f"\n📦 Batch Test:")
        print(f"   Resumes Processed: {batch_test['total_resumes']}")
        print(f"   Average Time per Resume: {batch_test['average_time']:.2f}s")
        print(f"   Total Matches Analyzed: {batch_test['total_matches']}")
        
        concurrent = report['test_results']['concurrent_test']
        print(f"\n🔄 Concurrent Operations:")
        print(f"   Concurrent Tasks: {concurrent['total_concurrent_tasks']}")
        print(f"   Success Rate: {concurrent['successful_tasks']}/{concurrent['total_concurrent_tasks']}")
        print(f"   Average Time per Task: {concurrent['average_time_per_task']}s")
        
        ranking = report['test_results']['ranking_validation']
        print(f"\n✅ Ranking Accuracy:")
        print(f"   Accuracy: {ranking['accuracy_percentage']}%")
        print(f"   Average Rank: {ranking['average_rank']:.1f}")
        
        print("\n" + "=" * 60)


async def main():
    """Run the performance tests"""
    test = LargeScaleMatchingTest()
    await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
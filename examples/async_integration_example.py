"""
Example demonstrating async integration for high-performance HR matching.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import json

from agents.async_orchestrator_agent import AsyncOrchestratorAgent
from agents.async_resume_parser_agent import AsyncResumeParserAgent
from tools.async_vector_store import get_async_vector_store
from tools.async_web_scraper import get_async_scraper
from tools.async_document_loaders import AsyncDocumentLoader


async def example_async_resume_processing():
    """Example: Process multiple resumes asynchronously"""
    print("=== Async Resume Processing Example ===\n")
    
    # Initialize components
    resume_parser = AsyncResumeParserAgent()
    vector_store = await get_async_vector_store()
    
    # Simulate multiple resume files
    resume_files = [
        "uploads/resumes/resume1.pdf",
        "uploads/resumes/resume2.docx",
        "uploads/resumes/resume3.txt"
    ]
    
    start_time = time.time()
    
    # Parse all resumes concurrently
    parse_tasks = []
    for file_path in resume_files:
        task = resume_parser.process({"file_path": file_path})
        parse_tasks.append(task)
    
    print(f"Parsing {len(resume_files)} resumes concurrently...")
    results = await asyncio.gather(*parse_tasks, return_exceptions=True)
    
    # Store successful parses in vector store
    storage_tasks = []
    successful_resumes = []
    
    for i, result in enumerate(results):
        if isinstance(result, dict) and result.get("success"):
            resume = result["resume"]
            successful_resumes.append(resume)
            storage_tasks.append(vector_store.add_resume(resume))
            print(f"✓ Successfully parsed: {resume['name']}")
        else:
            print(f"✗ Failed to parse resume {i+1}: {result}")
    
    # Store all resumes concurrently
    if storage_tasks:
        print(f"\nStoring {len(storage_tasks)} resumes in vector database...")
        await asyncio.gather(*storage_tasks)
    
    elapsed_time = time.time() - start_time
    print(f"\nProcessed {len(resume_files)} resumes in {elapsed_time:.2f} seconds")
    print(f"Rate: {len(resume_files)/elapsed_time:.2f} resumes/second")
    
    return successful_resumes


async def example_async_job_matching():
    """Example: Match resumes against multiple positions asynchronously"""
    print("\n\n=== Async Job Matching Example ===\n")
    
    vector_store = await get_async_vector_store()
    
    # Sample resume and position data
    sample_resume = """
    John Smith - Senior Software Engineer
    10+ years experience in distributed systems and cloud architecture.
    Skills: Python, Go, Kubernetes, AWS, Microservices, Docker
    Location: San Francisco, CA
    """
    
    sample_positions = [
        "Senior Backend Engineer - Python, AWS, Microservices",
        "Cloud Architect - Kubernetes, Docker, AWS",
        "Full Stack Developer - React, Python, PostgreSQL",
        "DevOps Engineer - Kubernetes, CI/CD, Terraform"
    ]
    
    start_time = time.time()
    
    # Search for similar positions
    print("Searching for matching positions...")
    matches = await vector_store.search_similar_positions(sample_resume, k=10)
    
    # Perform detailed matching for top positions
    detailed_match_tasks = []
    for match in matches[:5]:
        # Simulate detailed analysis
        task = analyze_match_details(sample_resume, match)
        detailed_match_tasks.append(task)
    
    detailed_results = await asyncio.gather(*detailed_match_tasks)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nFound {len(matches)} potential matches in {elapsed_time:.2f} seconds")
    print("\nTop Matches:")
    for i, (match, details) in enumerate(zip(matches[:5], detailed_results)):
        print(f"{i+1}. Position: {match.get('title', 'Unknown')}")
        print(f"   Score: {match.get('similarity_score', 0):.2%}")
        print(f"   Details: {details}")
    
    return matches


async def analyze_match_details(resume: str, position: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze match details (placeholder for detailed analysis)"""
    await asyncio.sleep(0.1)  # Simulate processing
    return {
        "skill_overlap": 0.85,
        "experience_match": 0.90,
        "location_match": 1.0
    }


async def example_salary_research():
    """Example: Concurrent salary research across multiple sources"""
    print("\n\n=== Async Salary Research Example ===\n")
    
    scraper = await get_async_scraper()
    
    # Research parameters
    job_titles = [
        "Software Engineer",
        "Senior Software Engineer",
        "Staff Software Engineer",
        "Principal Engineer"
    ]
    
    locations = [
        "San Francisco, CA",
        "New York, NY",
        "Seattle, WA",
        "Austin, TX"
    ]
    
    start_time = time.time()
    
    # Create all research tasks
    research_tasks = []
    for title in job_titles:
        for location in locations[:2]:  # Limit for demo
            task = scraper.aggregate_salary_data(
                job_title=title,
                location=location,
                experience_years=10
            )
            research_tasks.append(task)
    
    print(f"Researching salaries for {len(research_tasks)} job-location combinations...")
    results = await asyncio.gather(*research_tasks, return_exceptions=True)
    
    elapsed_time = time.time() - start_time
    
    # Process results
    salary_insights = []
    for i, result in enumerate(results):
        if isinstance(result, dict) and "aggregated_salary" in result:
            job_idx = i // 2
            loc_idx = i % 2
            
            insight = {
                "job_title": job_titles[job_idx],
                "location": locations[loc_idx],
                "average_salary": result["aggregated_salary"].get("average", 0),
                "confidence": result.get("confidence_score", 0)
            }
            salary_insights.append(insight)
    
    print(f"\nCompleted {len(research_tasks)} salary researches in {elapsed_time:.2f} seconds")
    print(f"Rate: {len(research_tasks)/elapsed_time:.2f} researches/second")
    
    print("\nSalary Insights:")
    for insight in salary_insights:
        print(f"- {insight['job_title']} in {insight['location']}: "
              f"${insight['average_salary']:,} (confidence: {insight['confidence']:.0%})")
    
    return salary_insights


async def example_bulk_processing():
    """Example: Bulk processing with progress tracking"""
    print("\n\n=== Async Bulk Processing Example ===\n")
    
    orchestrator = AsyncOrchestratorAgent()
    
    # Simulate bulk data
    num_resumes = 50
    num_positions = 20
    
    print(f"Processing {num_resumes} resumes against {num_positions} positions...")
    print(f"Total comparisons: {num_resumes * num_positions:,}")
    
    start_time = time.time()
    
    # Create progress tracking
    completed = 0
    total = num_resumes * num_positions
    
    async def process_with_progress(resume_id: str, position_id: str):
        nonlocal completed
        result = await orchestrator._match_pair(resume_id, position_id)
        completed += 1
        if completed % 100 == 0:
            progress = completed / total * 100
            print(f"Progress: {completed}/{total} ({progress:.1f}%)")
        return result
    
    # Create all matching tasks
    tasks = []
    for i in range(num_resumes):
        for j in range(num_positions):
            task = process_with_progress(f"resume_{i}", f"position_{j}")
            tasks.append(task)
    
    # Process in batches to avoid overwhelming the system
    batch_size = 100
    all_results = []
    
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        all_results.extend(batch_results)
    
    elapsed_time = time.time() - start_time
    
    # Filter and sort results
    valid_matches = [r for r in all_results if isinstance(r, dict) and r.get("score", 0) > 0.7]
    valid_matches.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"\nCompleted {total} comparisons in {elapsed_time:.2f} seconds")
    print(f"Rate: {total/elapsed_time:.2f} comparisons/second")
    print(f"Found {len(valid_matches)} high-quality matches (score > 0.7)")
    
    return valid_matches[:10]  # Return top 10


async def example_real_time_matching():
    """Example: Real-time matching with streaming results"""
    print("\n\n=== Real-time Matching Example ===\n")
    
    vector_store = await get_async_vector_store()
    
    # Simulate incoming resume
    new_resume = {
        "id": "realtime_001",
        "name": "Jane Doe",
        "skills": [
            {"name": "Python", "level": "Expert"},
            {"name": "Machine Learning", "level": "Advanced"},
            {"name": "TensorFlow", "level": "Intermediate"}
        ],
        "total_experience_years": 7,
        "location": "San Francisco, CA",
        "summary": "ML Engineer with expertise in deep learning and NLP"
    }
    
    print("New resume received. Starting real-time matching...")
    
    # Add to vector store
    await vector_store.add_resume(new_resume)
    
    # Create resume text for matching
    resume_text = f"{new_resume['summary']} Skills: {', '.join([s['name'] for s in new_resume['skills']])}"
    
    # Stream matching results
    async def stream_matches():
        # Initial quick matches
        quick_matches = await vector_store.search_similar_positions(resume_text, k=5)
        
        for match in quick_matches:
            yield {
                "type": "quick_match",
                "data": match,
                "timestamp": time.time()
            }
        
        # Detailed analysis (simulated)
        await asyncio.sleep(0.5)
        
        for match in quick_matches[:3]:
            detailed = await analyze_match_details(resume_text, match)
            yield {
                "type": "detailed_match",
                "data": {**match, "details": detailed},
                "timestamp": time.time()
            }
    
    # Process streaming results
    print("\nStreaming match results:")
    async for result in stream_matches():
        if result["type"] == "quick_match":
            print(f"  [Quick] Position: {result['data'].get('title', 'Unknown')} "
                  f"(Score: {result['data'].get('similarity_score', 0):.2%})")
        else:
            print(f"  [Detailed] Position: {result['data'].get('title', 'Unknown')} "
                  f"- Skills: {result['data']['details']['skill_overlap']:.2%}, "
                  f"Experience: {result['data']['details']['experience_match']:.2%}")
    
    print("\nReal-time matching completed!")


async def main():
    """Run all async examples"""
    print("=" * 60)
    print("HR Matcher Async Integration Examples")
    print("=" * 60)
    
    try:
        # Run examples
        resumes = await example_async_resume_processing()
        matches = await example_async_job_matching()
        salaries = await example_salary_research()
        bulk_results = await example_bulk_processing()
        await example_real_time_matching()
        
        # Save example results
        output_dir = Path("output/async_examples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "processed_resumes": len(resumes) if isinstance(resumes, list) else 0,
            "found_matches": len(matches) if isinstance(matches, list) else 0,
            "salary_insights": len(salaries) if isinstance(salaries, list) else 0,
            "bulk_matches": len(bulk_results) if isinstance(bulk_results, list) else 0
        }
        
        with open(output_dir / "example_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n\nExample results saved to {output_dir / 'example_results.json'}")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
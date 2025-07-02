"""
Performance tests comparing sync vs async implementations.
"""

import asyncio
import time
from pathlib import Path
import random
import string
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd

from tools.vector_store import VectorStoreManager
from tools.async_vector_store import AsyncVectorStoreManager
from tools.web_scraper import SalaryWebScraper
from tools.async_web_scraper import AsyncSalaryWebScraper
from agents.resume_parser_agent import ResumeParserAgent
from agents.async_resume_parser_agent import AsyncResumeParserAgent


def generate_mock_resume(index: int) -> Dict[str, Any]:
    """Generate mock resume data for testing"""
    return {
        "id": f"resume_{index}",
        "name": f"Test User {index}",
        "email": f"user{index}@example.com",
        "total_experience_years": random.randint(1, 15),
        "skills": [
            {"name": skill, "level": "Expert"}
            for skill in random.sample(
                ["Python", "Java", "JavaScript", "Go", "SQL", "AWS", "Docker", "Kubernetes"],
                k=random.randint(3, 6)
            )
        ],
        "location": random.choice(["San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA"]),
        "summary": f"Experienced professional with {random.randint(5, 15)} years in software development"
    }


def generate_mock_position(index: int) -> Dict[str, Any]:
    """Generate mock position data for testing"""
    return {
        "id": f"position_{index}",
        "title": random.choice(["Software Engineer", "Senior Developer", "Tech Lead", "Full Stack Engineer"]),
        "department": "Engineering",
        "location": random.choice(["San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA"]),
        "description": "Looking for experienced software engineers",
        "experience_level": random.choice(["Junior", "Mid", "Senior"]),
        "min_experience_years": random.randint(2, 10),
        "required_skills": random.sample(
            ["Python", "Java", "JavaScript", "Go", "SQL", "AWS", "Docker", "Kubernetes"],
            k=random.randint(3, 5)
        )
    }


async def test_vector_store_performance():
    """Test sync vs async vector store performance"""
    print("\n=== Vector Store Performance Test ===")
    
    # Generate test data
    num_items = 100
    resumes = [generate_mock_resume(i) for i in range(num_items)]
    
    # Test sync implementation
    sync_store = VectorStoreManager()
    start_time = time.time()
    
    for resume in resumes:
        sync_store.add_resume(resume)
    
    sync_time = time.time() - start_time
    print(f"Sync Vector Store: {num_items} resumes in {sync_time:.2f} seconds")
    print(f"Rate: {num_items/sync_time:.2f} resumes/second")
    
    # Test async implementation
    async_store = AsyncVectorStoreManager()
    start_time = time.time()
    
    await async_store.batch_add_resumes(resumes, batch_size=10)
    
    async_time = time.time() - start_time
    print(f"\nAsync Vector Store: {num_items} resumes in {async_time:.2f} seconds")
    print(f"Rate: {num_items/async_time:.2f} resumes/second")
    print(f"Speedup: {sync_time/async_time:.2f}x")
    
    return {"sync": sync_time, "async": async_time, "items": num_items}


async def test_web_scraper_performance():
    """Test sync vs async web scraper performance"""
    print("\n\n=== Web Scraper Performance Test ===")
    
    # Test parameters
    job_titles = ["Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer"]
    locations = ["San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA"]
    
    # Test sync implementation
    sync_scraper = SalaryWebScraper()
    start_time = time.time()
    
    sync_results = []
    for job in job_titles:
        for location in locations:
            result = sync_scraper.search_glassdoor_salaries(job, location)
            sync_results.append(result)
    
    sync_time = time.time() - start_time
    num_requests = len(job_titles) * len(locations)
    print(f"Sync Web Scraper: {num_requests} requests in {sync_time:.2f} seconds")
    print(f"Rate: {num_requests/sync_time:.2f} requests/second")
    
    # Test async implementation
    async_scraper = AsyncSalaryWebScraper()
    start_time = time.time()
    
    tasks = []
    for job in job_titles:
        for location in locations:
            task = async_scraper.search_glassdoor_salaries(job, location)
            tasks.append(task)
    
    async_results = await asyncio.gather(*tasks)
    
    async_time = time.time() - start_time
    print(f"\nAsync Web Scraper: {num_requests} requests in {async_time:.2f} seconds")
    print(f"Rate: {num_requests/async_time:.2f} requests/second")
    print(f"Speedup: {sync_time/async_time:.2f}x")
    
    return {"sync": sync_time, "async": async_time, "items": num_requests}


async def test_batch_matching_performance():
    """Test batch matching performance"""
    print("\n\n=== Batch Matching Performance Test ===")
    
    # Generate test data
    num_resumes = 50
    num_positions = 20
    
    resumes = [generate_mock_resume(i) for i in range(num_resumes)]
    positions = [generate_mock_position(i) for i in range(num_positions)]
    
    # Initialize stores
    async_store = AsyncVectorStoreManager()
    
    # Add data to store
    print("Adding test data to vector store...")
    await async_store.batch_add_resumes(resumes, batch_size=10)
    await async_store.batch_add_positions(positions, batch_size=10)
    
    # Test matching performance
    start_time = time.time()
    
    # Simulate batch matching
    total_matches = num_resumes * num_positions
    match_tasks = []
    
    for resume in resumes[:10]:  # Test with subset
        resume_text = f"{resume['summary']} Skills: {', '.join([s['name'] for s in resume['skills']])}"
        task = async_store.search_similar_positions(resume_text, k=5)
        match_tasks.append(task)
    
    match_results = await asyncio.gather(*match_tasks)
    
    async_time = time.time() - start_time
    num_searches = len(match_tasks)
    
    print(f"\nAsync Batch Matching: {num_searches} searches in {async_time:.2f} seconds")
    print(f"Rate: {num_searches/async_time:.2f} searches/second")
    
    return {"async": async_time, "items": num_searches}


async def test_concurrent_operations():
    """Test multiple concurrent operations"""
    print("\n\n=== Concurrent Operations Test ===")
    
    # Mix of different operations
    async_store = AsyncVectorStoreManager()
    async_scraper = AsyncSalaryWebScraper()
    
    start_time = time.time()
    
    # Create mixed workload
    tasks = []
    
    # Add vector store operations
    for i in range(20):
        resume = generate_mock_resume(i)
        tasks.append(async_store.add_resume(resume))
    
    # Add scraper operations
    for job in ["Software Engineer", "Data Scientist"]:
        for location in ["San Francisco, CA", "New York, NY"]:
            tasks.append(async_scraper.aggregate_salary_data(job, location))
    
    # Add search operations
    for i in range(10):
        tasks.append(async_store.search_similar_positions(f"Python developer with {i} years experience", k=3))
    
    # Execute all concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    successful = sum(1 for r in results if not isinstance(r, Exception))
    
    print(f"Executed {len(tasks)} mixed operations in {total_time:.2f} seconds")
    print(f"Success rate: {successful}/{len(tasks)} ({successful/len(tasks)*100:.1f}%)")
    print(f"Rate: {len(tasks)/total_time:.2f} operations/second")
    
    return {"time": total_time, "operations": len(tasks), "successful": successful}


def plot_performance_results(results: Dict[str, Any]):
    """Plot performance comparison results"""
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Async vs Sync Performance Comparison', fontsize=16)
    
    # Vector Store Performance
    ax1 = axes[0, 0]
    if "vector_store" in results:
        data = results["vector_store"]
        ax1.bar(['Sync', 'Async'], [data['sync'], data['async']])
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title(f'Vector Store: {data["items"]} items')
        ax1.text(0, data['sync'], f"{data['sync']:.2f}s", ha='center', va='bottom')
        ax1.text(1, data['async'], f"{data['async']:.2f}s", ha='center', va='bottom')
    
    # Web Scraper Performance
    ax2 = axes[0, 1]
    if "web_scraper" in results:
        data = results["web_scraper"]
        ax2.bar(['Sync', 'Async'], [data['sync'], data['async']])
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title(f'Web Scraper: {data["items"]} requests')
        ax2.text(0, data['sync'], f"{data['sync']:.2f}s", ha='center', va='bottom')
        ax2.text(1, data['async'], f"{data['async']:.2f}s", ha='center', va='bottom')
    
    # Speedup Chart
    ax3 = axes[1, 0]
    speedups = []
    labels = []
    
    if "vector_store" in results:
        speedups.append(results["vector_store"]["sync"] / results["vector_store"]["async"])
        labels.append("Vector Store")
    
    if "web_scraper" in results:
        speedups.append(results["web_scraper"]["sync"] / results["web_scraper"]["async"])
        labels.append("Web Scraper")
    
    ax3.bar(labels, speedups, color=['green', 'blue'])
    ax3.set_ylabel('Speedup Factor')
    ax3.set_title('Async Speedup vs Sync')
    ax3.axhline(y=1, color='red', linestyle='--', label='No speedup')
    
    for i, v in enumerate(speedups):
        ax3.text(i, v, f"{v:.2f}x", ha='center', va='bottom')
    
    # Operations per Second
    ax4 = axes[1, 1]
    if "concurrent" in results:
        data = results["concurrent"]
        rate = data["operations"] / data["time"]
        ax4.bar(['Concurrent Ops'], [rate])
        ax4.set_ylabel('Operations/Second')
        ax4.set_title('Concurrent Operations Performance')
        ax4.text(0, rate, f"{rate:.2f} ops/s", ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("output/performance_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "async_performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nPerformance plot saved to {output_dir / 'async_performance_comparison.png'}")


async def main():
    """Run all performance tests"""
    print("=" * 60)
    print("HR Matcher Async Performance Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    try:
        results["vector_store"] = await test_vector_store_performance()
    except Exception as e:
        print(f"Vector store test failed: {e}")
    
    try:
        results["web_scraper"] = await test_web_scraper_performance()
    except Exception as e:
        print(f"Web scraper test failed: {e}")
    
    try:
        results["batch_matching"] = await test_batch_matching_performance()
    except Exception as e:
        print(f"Batch matching test failed: {e}")
    
    try:
        results["concurrent"] = await test_concurrent_operations()
    except Exception as e:
        print(f"Concurrent operations test failed: {e}")
    
    # Summary
    print("\n\n" + "=" * 60)
    print("PERFORMANCE TEST SUMMARY")
    print("=" * 60)
    
    total_speedup = []
    
    if "vector_store" in results:
        speedup = results["vector_store"]["sync"] / results["vector_store"]["async"]
        total_speedup.append(speedup)
        print(f"Vector Store Speedup: {speedup:.2f}x")
    
    if "web_scraper" in results:
        speedup = results["web_scraper"]["sync"] / results["web_scraper"]["async"]
        total_speedup.append(speedup)
        print(f"Web Scraper Speedup: {speedup:.2f}x")
    
    if total_speedup:
        avg_speedup = sum(total_speedup) / len(total_speedup)
        print(f"\nAverage Speedup: {avg_speedup:.2f}x")
    
    # Generate plots
    try:
        plot_performance_results(results)
    except Exception as e:
        print(f"Failed to generate plots: {e}")
    
    # Save results to JSON
    import json
    output_dir = Path("output/performance_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "performance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'performance_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
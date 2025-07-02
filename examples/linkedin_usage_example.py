#!/usr/bin/env python3
"""
Example usage of LinkedIn integration for job matching.

This script demonstrates how to use the LinkedIn integration to:
1. Fetch jobs from specific companies
2. Match resumes against job positions
3. Compare opportunities across multiple companies
4. Get market insights
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

from agents.linkedin_agent import LinkedInAgent
from agents.orchestrator_agent import OrchestratorAgent


# Sample resume for testing
SAMPLE_RESUME = """
Jane Smith
jane.smith@email.com | +1-555-123-4567 | San Francisco, CA
linkedin.com/in/janesmith | github.com/janesmith

SUMMARY
Senior Software Engineer with 7+ years of experience building scalable distributed systems
and leading cross-functional teams. Expertise in Python, microservices architecture, and
cloud technologies. Passionate about mentoring and driving technical excellence.

TECHNICAL SKILLS
Languages: Python, Go, JavaScript, TypeScript, Java
Frameworks: Django, FastAPI, Flask, React, Node.js
Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
Cloud/DevOps: AWS (EC2, S3, Lambda, RDS), Docker, Kubernetes, Terraform, CI/CD
Tools: Git, Jenkins, JIRA, Datadog, Grafana

EXPERIENCE
Senior Software Engineer | TechCorp Inc. | San Francisco, CA | 2019 - Present
• Led migration of monolithic application to microservices, improving scalability by 300%
• Designed and implemented real-time data pipeline processing 10M+ events daily
• Mentored team of 5 junior engineers, conducting code reviews and technical training
• Reduced infrastructure costs by 40% through optimization and auto-scaling strategies

Software Engineer | StartupXYZ | San Francisco, CA | 2016 - 2019
• Built RESTful APIs serving 1M+ daily active users with 99.9% uptime
• Implemented caching strategies reducing database load by 60%
• Collaborated with product team to deliver features ahead of schedule
• Introduced automated testing increasing code coverage from 40% to 85%

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2012 - 2016
GPA: 3.8/4.0, Dean's List

CERTIFICATIONS
• AWS Certified Solutions Architect - Associate
• Certified Kubernetes Administrator (CKA)
"""


async def example_single_company_analysis():
    """Example: Analyze opportunities at a single company."""
    print("=== SINGLE COMPANY ANALYSIS ===\n")
    
    # Initialize LinkedIn agent
    agent = LinkedInAgent(
        name="LinkedIn Specialist",
        mcp_server_url="http://localhost:8002"
    )
    
    # Analyze opportunities at a specific company
    result = await agent.analyze_company_opportunities(
        company_name="Google",
        resume_text=SAMPLE_RESUME,
        location="San Francisco, CA",
        preferences={
            "job_type": "Full-time",
            "experience_level": "Senior"
        }
    )
    
    print(f"Company: {result.get('company')}")
    print(f"\nAnalysis Summary:")
    summary = result.get('summary', {})
    print(f"  Total Positions: {summary.get('total_positions', 'N/A')}")
    print(f"  Match Rate: {summary.get('match_rate', 'N/A')}%")
    print(f"  Top Skills Required: {', '.join(summary.get('top_skills', []))}")
    
    print("\nRecommendations:")
    for rec in summary.get('recommendations', [])[:3]:
        print(f"  • {rec}")


async def example_multi_company_comparison():
    """Example: Compare opportunities across multiple companies."""
    print("\n\n=== MULTI-COMPANY COMPARISON ===\n")
    
    agent = LinkedInAgent(
        name="LinkedIn Specialist",
        mcp_server_url="http://localhost:8002"
    )
    
    # Compare across top tech companies
    companies = ["Google", "Meta", "Amazon", "Microsoft", "Apple"]
    
    result = await agent.compare_companies(
        resume_text=SAMPLE_RESUME,
        company_list=companies,
        criteria={
            "focus_on": ["match_quality", "growth_opportunities", "tech_stack_alignment"],
            "location": "San Francisco Bay Area"
        }
    )
    
    print(f"Companies Analyzed: {', '.join(result.get('companies_analyzed', []))}")
    
    print("\nCompany Rankings:")
    rankings = result.get('rankings', {})
    for company, rank in sorted(rankings.items(), key=lambda x: x[1]):
        print(f"  {rank}. {company}")
    
    print("\nBest Matches:")
    for match in result.get('best_matches', [])[:5]:
        print(f"  • {match}")


async def example_market_insights():
    """Example: Get market insights for a specific role."""
    print("\n\n=== MARKET INSIGHTS ===\n")
    
    agent = LinkedInAgent(
        name="LinkedIn Specialist",
        mcp_server_url="http://localhost:8002"
    )
    
    # Get insights for Senior Software Engineer role
    result = await agent.get_market_insights(
        role="Senior Software Engineer",
        companies=["Google", "Meta", "Amazon", "Microsoft"],
        location="San Francisco Bay Area"
    )
    
    print(f"Role: {result.get('role')}")
    
    key_findings = result.get('key_findings', {})
    
    print("\nRequired Skills in Demand:")
    for skill in key_findings.get('required_skills', []):
        print(f"  • {skill}")
    
    print("\nExperience Levels:")
    for level in key_findings.get('experience_levels', []):
        print(f"  • {level}")
    
    print("\nMarket Trends:")
    for trend in key_findings.get('trends', [])[:3]:
        print(f"  • {trend}")


async def example_job_search_optimization():
    """Example: Optimize job search strategy."""
    print("\n\n=== JOB SEARCH OPTIMIZATION ===\n")
    
    agent = LinkedInAgent(
        name="LinkedIn Specialist",
        mcp_server_url="http://localhost:8002"
    )
    
    # Get optimized job search strategy
    target_companies = ["Google", "Meta", "Airbnb", "Stripe", "Uber"]
    
    result = await agent.optimize_job_search(
        resume_text=SAMPLE_RESUME,
        target_companies=target_companies,
        target_role="Senior Software Engineer"
    )
    
    print("Job Search Strategy:\n")
    
    print("Priority Companies:")
    for company in result.get('priority_companies', [])[:3]:
        print(f"  • {company}")
    
    print("\nPriority Skills to Highlight:")
    for skill in result.get('priority_skills', [])[:5]:
        print(f"  • {skill}")
    
    print("\nAction Items:")
    for action in result.get('action_items', [])[:5]:
        print(f"  • {action}")


async def example_orchestrated_job_search():
    """Example: Use orchestrator for comprehensive job search."""
    print("\n\n=== ORCHESTRATED JOB SEARCH ===\n")
    
    # Use orchestrator to coordinate multiple agents
    orchestrator = OrchestratorAgent(
        linkedin_mcp_url="http://localhost:8002",
        kaggle_mcp_url="http://localhost:8000"
    )
    
    # Comprehensive job search request
    request = """
    I'm a Senior Software Engineer looking for new opportunities.
    Please help me:
    1. Find and analyze positions at Google, Meta, and Amazon
    2. Match my resume against these positions
    3. Identify skill gaps and provide recommendations
    4. Suggest an application strategy
    
    Focus on senior-level positions in the San Francisco Bay Area.
    """
    
    result = await orchestrator.process(request, {"resume": SAMPLE_RESUME})
    
    print("Orchestrated Analysis Complete!")
    print(f"\nKey Findings:")
    print(result)


async def example_bulk_analysis():
    """Example: Analyze opportunities across many companies."""
    print("\n\n=== BULK COMPANY ANALYSIS ===\n")
    
    from tools.linkedin_integration import LinkedInBulkAnalysisTool
    
    # Initialize bulk analysis tool
    tool = LinkedInBulkAnalysisTool(
        mcp_server_url="http://localhost:8002"
    )
    
    # Analyze across multiple companies
    companies = [
        "Google", "Meta", "Amazon", "Microsoft", "Apple",
        "Netflix", "Airbnb", "Uber", "Lyft", "Stripe",
        "Salesforce", "Adobe", "Oracle", "VMware", "Nvidia"
    ]
    
    result_json = await tool._arun(
        resume_text=SAMPLE_RESUME,
        company_list=companies[:10],  # First 10 companies
        location="San Francisco Bay Area",
        min_match_score=0.6
    )
    
    print(result_json)


async def example_save_results():
    """Example: Save analysis results to file."""
    print("\n\n=== SAVING RESULTS ===\n")
    
    agent = LinkedInAgent(
        name="LinkedIn Specialist",
        mcp_server_url="http://localhost:8002"
    )
    
    # Perform analysis
    companies = ["Google", "Meta", "Amazon"]
    results = {}
    
    for company in companies:
        print(f"Analyzing {company}...")
        result = await agent.analyze_company_opportunities(
            company_name=company,
            resume_text=SAMPLE_RESUME,
            location="San Francisco, CA"
        )
        results[company] = result
    
    # Save results
    output_dir = Path("output/linkedin_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    with open(output_dir / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary report
    with open(output_dir / "summary_report.txt", "w") as f:
        f.write("LinkedIn Job Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for company, result in results.items():
            f.write(f"Company: {company}\n")
            summary = result.get('summary', {})
            f.write(f"  Positions: {summary.get('total_positions', 'N/A')}\n")
            f.write(f"  Match Rate: {summary.get('match_rate', 'N/A')}%\n")
            f.write(f"  Top Skills: {', '.join(summary.get('top_skills', []))}\n")
            f.write("\n")
    
    print(f"Results saved to {output_dir}")


def print_section_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60 + "\n")


async def main():
    """Run all examples."""
    print_section_header("LinkedIn Integration Examples")
    
    try:
        # Run examples
        await example_single_company_analysis()
        await example_multi_company_comparison()
        await example_market_insights()
        await example_job_search_optimization()
        
        # These require MCP servers to be running
        print("\n\nNote: The following examples require MCP servers to be running:")
        print("  1. python mcp_server/linkedin_jobs_server.py")
        print("  2. python mcp_server/advanced_resume_analyzer.py")
        print("  3. python mcp_server/kaggle_resume_server.py")
        
        # Uncomment to run with servers
        # await example_orchestrated_job_search()
        # await example_bulk_analysis()
        # await example_save_results()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the MCP servers are running if you're testing server-dependent features.")


if __name__ == "__main__":
    asyncio.run(main())
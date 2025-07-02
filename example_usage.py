import os
import asyncio
from agents.orchestrator_agent import OrchestratorAgent
from models.schemas import BatchMatchRequest

# Create example data directory
os.makedirs("example_data", exist_ok=True)

# Example resume content
example_resume = """
John Doe
Email: john.doe@email.com
Phone: (555) 123-4567
Location: San Francisco, CA

PROFESSIONAL SUMMARY
Experienced Software Engineer with 8 years of expertise in developing scalable web applications 
and distributed systems. Strong background in Python, cloud technologies, and machine learning. 
Proven track record of leading technical teams and delivering high-impact projects.

EXPERIENCE

Senior Software Engineer | Tech Innovations Inc. | San Francisco, CA | 2020 - Present
- Led development of microservices architecture serving 10M+ users
- Implemented machine learning models for recommendation system
- Mentored junior developers and conducted code reviews
- Technologies: Python, FastAPI, Docker, Kubernetes, AWS, PostgreSQL

Software Engineer | StartupXYZ | San Francisco, CA | 2017 - 2020
- Developed RESTful APIs and real-time data processing pipelines
- Optimized database queries reducing response time by 60%
- Collaborated with product team to define technical requirements
- Technologies: Python, Django, Redis, MongoDB, Apache Kafka

Junior Software Engineer | WebCorp | San Jose, CA | 2016 - 2017
- Built responsive web applications using modern frameworks
- Participated in agile development processes
- Technologies: Python, JavaScript, React, MySQL

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2016

SKILLS
Programming: Python, JavaScript, Go, SQL
Frameworks: FastAPI, Django, React, Vue.js
Cloud: AWS, GCP, Docker, Kubernetes
Databases: PostgreSQL, MongoDB, Redis
Machine Learning: TensorFlow, scikit-learn, PyTorch

CERTIFICATIONS
- AWS Certified Solutions Architect
- Google Cloud Professional Data Engineer

Expected Salary: $150,000 - $180,000
"""

example_job = """
Senior Software Engineer - Machine Learning Platform

Tech Giants Corp | San Francisco, CA | Hybrid

About the Role:
We are seeking an experienced Senior Software Engineer to join our Machine Learning Platform team. 
You will be responsible for building and maintaining scalable infrastructure that powers our 
AI/ML initiatives across the organization.

Responsibilities:
- Design and implement distributed systems for ML model training and deployment
- Build APIs and services for model serving at scale
- Collaborate with data scientists to productionize ML models
- Mentor junior engineers and promote best practices
- Participate in on-call rotation and ensure system reliability

Requirements:
- 5+ years of software engineering experience
- Strong proficiency in Python and at least one other language
- Experience with cloud platforms (AWS, GCP, or Azure)
- Knowledge of containerization and orchestration (Docker, Kubernetes)
- Experience with machine learning frameworks (TensorFlow, PyTorch)
- Bachelor's degree in Computer Science or related field

Preferred Qualifications:
- Experience with MLOps and model deployment pipelines
- Knowledge of distributed computing frameworks (Spark, Ray)
- Experience with real-time data processing
- Previous experience in a senior or lead role

Salary Range: $140,000 - $200,000
Benefits: Health insurance, 401k matching, equity, unlimited PTO
"""

async def create_example_files():
    """Create example resume and job files"""
    
    # Save example resume
    with open("example_data/john_doe_resume.txt", "w") as f:
        f.write(example_resume)
    
    # Save example job
    with open("example_data/senior_swe_ml_position.txt", "w") as f:
        f.write(example_job)
    
    print("✅ Created example files in example_data/")

async def test_single_match():
    """Test single resume-position matching"""
    print("\n🔍 Testing Single Match...")
    
    orchestrator = OrchestratorAgent()
    
    result = orchestrator.process_single_match(
        resume_path="example_data/john_doe_resume.txt",
        position_path="example_data/senior_swe_ml_position.txt",
        include_salary=True,
        include_aspirations=True
    )
    
    if result["success"]:
        match = result["match"]
        print(f"\n✅ Match Result:")
        print(f"  Overall Score: {match['overall_score']:.2%}")
        print(f"  Skill Match: {match['skill_match_score']:.2%}")
        print(f"  Experience Match: {match['experience_match_score']:.2%}")
        print(f"  Education Match: {match['education_match_score']:.2%}")
        
        print(f"\n💪 Strengths:")
        for strength in match['strengths'][:3]:
            print(f"  - {strength}")
        
        print(f"\n⚠️  Gaps:")
        for gap in match['gaps'][:3]:
            print(f"  - {gap}")
        
        print(f"\n💡 Recommendations:")
        for rec in match['recommendations'][:3]:
            print(f"  - {rec}")
        
        if "salary_research" in result:
            salary = result["salary_research"]["salary_research"]
            print(f"\n💰 Salary Research:")
            print(f"  Market Average: ${salary['market_average']:,}")
            print(f"  Market Range: ${salary['market_min']:,} - ${salary['market_max']:,}")
        
        if "aspiration_analysis" in result:
            aspirations = result["aspiration_analysis"]
            print(f"\n🎯 Career Aspirations:")
            for insight in aspirations["insights"][:3]:
                print(f"  - {insight}")
    else:
        print(f"❌ Error: {result['error']}")

async def test_api_endpoints():
    """Test API endpoints (requires API server running)"""
    import requests
    
    print("\n🌐 Testing API Endpoints...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Health Check: {health['status']}")
            print(f"  Ollama: {health['ollama_status']}")
            print(f"  Vector Store: {health['vector_store_status']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
        
        # Test salary research
        salary_params = {
            "position_title": "Senior Software Engineer",
            "location": "San Francisco",
            "experience_years": 8
        }
        
        response = requests.post(f"{base_url}/research/salary", params=salary_params)
        if response.status_code == 200:
            salary_data = response.json()
            print(f"\n✅ Salary Research API:")
            print(f"  Market Average: ${salary_data['salary_research']['market_average']:,}")
        else:
            print(f"❌ Salary research failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running with: python api/main.py")

async def main():
    """Run all examples"""
    print("🚀 HR Resume Matcher - Example Usage\n")
    
    # Create example files
    await create_example_files()
    
    # Test single match
    await test_single_match()
    
    # Test API endpoints
    await test_api_endpoints()
    
    print("\n✅ Example completed!")
    print("\n📝 To start the API server, run:")
    print("   python api/main.py")
    print("\n📝 To use with real data:")
    print("   1. Start Ollama: ollama serve")
    print("   2. Pull model: ollama pull llama3.2")
    print("   3. Start API: python api/main.py")
    print("   4. Upload resumes and positions via API")

if __name__ == "__main__":
    asyncio.run(main())
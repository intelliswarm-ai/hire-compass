import asyncio
import json
import os
from datetime import datetime

# Example usage of the Resume2Post MCP server

async def example_single_categorization():
    """Example: Categorize a single resume"""
    print("\n📄 Example 1: Single Resume Categorization")
    print("-" * 50)
    
    # This would be called through MCP client
    example_request = {
        "tool": "categorize_resume",
        "arguments": {
            "resume_path": "example_data/john_doe_resume.txt",
            "top_k": 5,
            "min_confidence": "medium",
            "filter_location": "San Francisco"
        }
    }
    
    print(f"Request: {json.dumps(example_request, indent=2)}")
    
    # Expected response format
    example_response = {
        "success": True,
        "resume_id": "resume_12345",
        "resume_name": "John Doe",
        "categorized_jobs": [
            {
                "job_id": "pos_001",
                "job_title": "Senior Software Engineer - ML Platform",
                "company": "Tech Giants Corp",
                "location": "San Francisco, CA",
                "semantic_score": 0.85,
                "feature_score": 0.82,
                "final_score": 0.84,
                "confidence": "high",
                "match_reasons": [
                    "Matching skills: python, machine learning, aws",
                    "Experience requirement met (8 years)",
                    "Location match"
                ]
            },
            {
                "job_id": "pos_002", 
                "job_title": "ML Engineer",
                "company": "AI Startup Inc",
                "location": "San Francisco, CA",
                "semantic_score": 0.78,
                "feature_score": 0.75,
                "final_score": 0.77,
                "confidence": "medium",
                "match_reasons": [
                    "Matching skills: tensorflow, python",
                    "Experience requirement met (8 years)"
                ]
            }
        ],
        "confidence_summary": {
            "high": 1,
            "medium": 3,
            "low": 1
        },
        "processing_time": 2.5
    }
    
    print(f"\nExpected Response: {json.dumps(example_response, indent=2)}")

async def example_batch_categorization():
    """Example: Batch categorize multiple resumes"""
    print("\n📚 Example 2: Batch Resume Categorization")
    print("-" * 50)
    
    example_request = {
        "tool": "batch_categorize_resumes",
        "arguments": {
            "resume_paths": [
                "example_data/john_doe_resume.txt",
                "example_data/jane_smith_resume.txt",
                "example_data/bob_johnson_resume.txt"
            ],
            "top_k": 3,
            "parallel": True
        }
    }
    
    print(f"Request: {json.dumps(example_request, indent=2)}")
    
    example_response = {
        "success": True,
        "total_resumes": 3,
        "processed": 3,
        "results": [
            {
                "resume_id": "resume_12345",
                "resume_name": "John Doe",
                "top_matches": [
                    {
                        "job_title": "Senior Software Engineer - ML",
                        "company": "Tech Corp",
                        "score": 0.84,
                        "confidence": "high"
                    }
                ],
                "total_matches": 5
            }
        ]
    }
    
    print(f"\nExpected Response: {json.dumps(example_response, indent=2)}")

async def example_model_training():
    """Example: Train the categorizer model"""
    print("\n🎓 Example 3: Model Training")
    print("-" * 50)
    
    # Create example training data
    training_data = [
        {
            "resume_data": {
                "id": "resume_001",
                "name": "John Doe",
                "skills": [{"name": "Python"}, {"name": "ML"}],
                "total_experience_years": 8,
                "raw_text": "Experienced ML engineer..."
            },
            "job_data": {
                "id": "job_001",
                "title": "Senior ML Engineer",
                "required_skills": ["Python", "TensorFlow"],
                "min_experience_years": 5
            },
            "overall_score": 0.85
        }
    ]
    
    # Save training data
    os.makedirs("example_data", exist_ok=True)
    with open("example_data/training_history.json", "w") as f:
        json.dump(training_data, f)
    
    example_request = {
        "tool": "train_categorizer",
        "arguments": {
            "match_history_file": "example_data/training_history.json",
            "min_score_threshold": 0.7
        }
    }
    
    print(f"Request: {json.dumps(example_request, indent=2)}")

async def example_match_explanation():
    """Example: Get match explanation"""
    print("\n🔍 Example 4: Match Explanation")
    print("-" * 50)
    
    example_request = {
        "tool": "explain_match",
        "arguments": {
            "resume_path": "example_data/john_doe_resume.txt",
            "job_id": "pos_001"
        }
    }
    
    print(f"Request: {json.dumps(example_request, indent=2)}")
    
    example_response = {
        "success": True,
        "explanation": {
            "resume_id": "resume_12345",
            "job_id": "pos_001",
            "match_factors": {
                "skills": {
                    "matched": ["Python", "Machine Learning", "AWS"],
                    "missing": ["Kubernetes", "Go"],
                    "score": 0.75
                },
                "experience": {
                    "required": "5+ years",
                    "candidate": "8 years",
                    "score": 0.9
                },
                "education": {
                    "meets_requirement": True,
                    "score": 1.0
                },
                "location": {
                    "compatible": True,
                    "score": 1.0
                }
            },
            "overall_compatibility": 0.85,
            "recommendation": "Strong candidate - recommend interview"
        }
    }
    
    print(f"\nExpected Response: {json.dumps(example_response, indent=2)}")

async def example_update_weights():
    """Example: Update model weights"""
    print("\n⚖️ Example 5: Update Model Weights")
    print("-" * 50)
    
    example_request = {
        "tool": "update_model_weights",
        "arguments": {
            "semantic_weight": 0.7,
            "feature_weight": 0.3
        }
    }
    
    print(f"Request: {json.dumps(example_request, indent=2)}")
    
    example_response = {
        "success": True,
        "semantic_weight": 0.7,
        "feature_weight": 0.3,
        "message": "Model weights updated successfully"
    }
    
    print(f"\nExpected Response: {json.dumps(example_response, indent=2)}")

async def main():
    """Run all examples"""
    print("🚀 Resume2Post MCP Server - Usage Examples")
    print("=" * 50)
    
    await example_single_categorization()
    await example_batch_categorization()
    await example_model_training()
    await example_match_explanation()
    await example_update_weights()
    
    print("\n📝 Integration with Claude Desktop:")
    print("-" * 50)
    print("1. Add to Claude Desktop config:")
    print(json.dumps({
        "mcpServers": {
            "resume2post": {
                "command": "python",
                "args": ["mcp_server/server.py"],
                "cwd": os.getcwd()
            }
        }
    }, indent=2))
    
    print("\n2. Example conversation with Claude:")
    print("   'Please categorize the resume at /path/to/resume.pdf to our open positions'")
    print("   'Show me the top 5 job matches for this candidate with high confidence'")
    print("   'Explain why this resume matches the Senior ML Engineer position'")
    
    print("\n✅ Examples completed!")

if __name__ == "__main__":
    asyncio.run(main())
import random
import json
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta
import uuid
from faker import Faker

fake = Faker()

class JobPositionGenerator:
    """Generate synthetic job positions for testing"""
    
    def __init__(self):
        self.departments = [
            "Engineering", "Data Science", "Product", "Design", 
            "Marketing", "Sales", "Operations", "Finance", "HR", "Legal"
        ]
        
        self.job_titles = {
            "Engineering": [
                "Software Engineer", "Senior Software Engineer", "Staff Engineer",
                "Principal Engineer", "Engineering Manager", "Technical Lead",
                "DevOps Engineer", "Frontend Developer", "Backend Developer",
                "Full Stack Developer", "Mobile Developer", "QA Engineer"
            ],
            "Data Science": [
                "Data Scientist", "Senior Data Scientist", "ML Engineer",
                "Data Analyst", "Business Analyst", "Data Engineer",
                "Research Scientist", "AI Engineer", "Analytics Manager"
            ],
            "Product": [
                "Product Manager", "Senior Product Manager", "Product Owner",
                "Technical Product Manager", "Product Designer", "UX Researcher"
            ],
            "Design": [
                "UI Designer", "UX Designer", "Product Designer",
                "Graphic Designer", "Design Lead", "Creative Director"
            ]
        }
        
        self.skills_pool = {
            "technical": [
                "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust",
                "React", "Angular", "Vue", "Node.js", "Django", "Flask",
                "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
                "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Kafka",
                "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
                "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy"
            ],
            "soft": [
                "Leadership", "Communication", "Problem Solving", "Teamwork",
                "Project Management", "Agile", "Scrum", "Mentoring",
                "Strategic Thinking", "Analytical Skills", "Creativity"
            ]
        }
        
        self.locations = [
            "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
            "Boston, MA", "Los Angeles, CA", "Chicago, IL", "Denver, CO",
            "Portland, OR", "Miami, FL", "Remote", "Hybrid - SF", "Hybrid - NYC"
        ]
        
        self.companies = [
            "TechCorp Inc", "DataVision Systems", "CloudScale Solutions",
            "AI Innovations", "Digital Dynamics", "FutureTech Labs",
            "Quantum Computing Co", "NextGen Software", "CyberSecure Inc",
            "Analytics Pro", "DevOps Masters", "Mobile First Inc"
        ]
    
    def generate_job_position(self, index: int) -> Dict[str, Any]:
        """Generate a single job position"""
        department = random.choice(self.departments)
        
        # Get appropriate job titles for department
        if department in self.job_titles:
            title = random.choice(self.job_titles[department])
        else:
            title = f"{department} Specialist"
        
        # Determine experience level based on title
        if "Senior" in title or "Staff" in title:
            exp_level = "senior"
            min_exp = random.randint(5, 8)
        elif "Principal" in title or "Lead" in title or "Manager" in title:
            exp_level = "lead"
            min_exp = random.randint(8, 12)
        elif "Junior" in title or "Entry" in title:
            exp_level = "entry"
            min_exp = random.randint(0, 2)
        else:
            exp_level = "mid"
            min_exp = random.randint(3, 5)
        
        max_exp = min_exp + random.randint(3, 5)
        
        # Generate skills
        num_required_skills = random.randint(5, 10)
        num_preferred_skills = random.randint(3, 7)
        
        all_tech_skills = random.sample(self.skills_pool["technical"], 
                                       min(num_required_skills + num_preferred_skills, 
                                           len(self.skills_pool["technical"])))
        
        required_skills = all_tech_skills[:num_required_skills]
        preferred_skills = all_tech_skills[num_required_skills:]
        
        # Add some soft skills
        soft_skills = random.sample(self.skills_pool["soft"], 3)
        preferred_skills.extend(soft_skills)
        
        # Generate salary range
        base_salary = {
            "entry": 80000,
            "mid": 120000,
            "senior": 160000,
            "lead": 200000
        }
        
        salary_min = base_salary.get(exp_level, 100000) + random.randint(-20000, 20000)
        salary_max = salary_min + random.randint(30000, 60000)
        
        # Generate description
        description = f"""
        We are looking for a talented {title} to join our {department} team at {random.choice(self.companies)}.
        
        In this role, you will be responsible for developing and maintaining cutting-edge solutions
        that impact millions of users. You'll work with a talented team of professionals in a 
        fast-paced, innovative environment.
        
        Key Responsibilities:
        - Design and implement scalable solutions
        - Collaborate with cross-functional teams
        - Mentor junior team members
        - Drive technical excellence and best practices
        - Participate in code reviews and architecture discussions
        
        What We Offer:
        - Competitive salary and equity
        - Comprehensive health benefits
        - Flexible work arrangements
        - Professional development opportunities
        - Cutting-edge technology stack
        """
        
        # Generate responsibilities
        responsibilities = [
            f"Design and develop high-quality {random.choice(['software', 'systems', 'applications'])}",
            f"Collaborate with {random.choice(['product', 'design', 'engineering'])} teams",
            "Write clean, maintainable, and efficient code",
            "Participate in code reviews and provide constructive feedback",
            f"Mentor junior {random.choice(['developers', 'engineers', 'team members'])}",
            "Contribute to technical documentation and best practices",
            "Troubleshoot and debug complex issues",
            "Stay up-to-date with industry trends and technologies"
        ]
        
        # Generate requirements
        requirements = [
            f"{min_exp}+ years of professional experience",
            f"Strong proficiency in {', '.join(required_skills[:3])}",
            "Bachelor's degree in Computer Science or related field",
            "Excellent problem-solving and analytical skills",
            "Strong communication and collaboration abilities",
            "Experience with agile development methodologies"
        ]
        
        position = {
            "id": f"pos_{str(uuid.uuid4())[:8]}_{index}",
            "title": title,
            "department": department,
            "company": random.choice(self.companies),
            "location": random.choice(self.locations),
            "work_mode": random.choice(["onsite", "remote", "hybrid"]),
            "description": description.strip(),
            "responsibilities": responsibilities,
            "requirements": requirements,
            "preferred_qualifications": [
                f"Experience with {', '.join(preferred_skills[:2])}",
                "Previous experience in a similar role",
                "Strong technical leadership skills"
            ],
            "required_skills": required_skills,
            "preferred_skills": preferred_skills,
            "experience_level": exp_level,
            "min_experience_years": min_exp,
            "max_experience_years": max_exp,
            "education_requirements": ["bachelors"] if random.random() > 0.3 else ["bachelors", "masters"],
            "salary_range_min": salary_min,
            "salary_range_max": salary_max,
            "posted_date": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
            "is_active": True,
            "application_deadline": (datetime.now() + timedelta(days=random.randint(30, 90))).isoformat()
        }
        
        return position
    
    def generate_positions(self, count: int = 300) -> List[Dict[str, Any]]:
        """Generate multiple job positions"""
        positions = []
        for i in range(count):
            position = self.generate_job_position(i)
            positions.append(position)
            
            # Add some variety in skills and requirements
            if i % 10 == 0:
                # Every 10th position, shuffle the skills pool
                random.shuffle(self.skills_pool["technical"])
        
        return positions
    
    def save_positions(self, positions: List[Dict[str, Any]], filename: str = "job_positions_300.json"):
        """Save positions to JSON file"""
        filepath = os.path.join("tests/data", filename)
        with open(filepath, 'w') as f:
            json.dump(positions, f, indent=2)
        print(f"✅ Saved {len(positions)} job positions to {filepath}")
        return filepath


class ResumeGenerator:
    """Generate synthetic resumes for testing"""
    
    def __init__(self):
        self.first_names = ["John", "Jane", "Michael", "Sarah", "David", "Emily", 
                           "Robert", "Lisa", "James", "Maria", "William", "Jennifer"]
        self.last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                          "Miller", "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor"]
        
        self.universities = [
            "MIT", "Stanford University", "UC Berkeley", "Carnegie Mellon",
            "Georgia Tech", "University of Washington", "UT Austin",
            "University of Illinois", "Columbia University", "Cornell University"
        ]
        
        self.degrees = [
            "Computer Science", "Software Engineering", "Data Science",
            "Information Systems", "Computer Engineering", "Mathematics",
            "Statistics", "Physics", "Electrical Engineering"
        ]
        
        self.job_position_generator = JobPositionGenerator()
    
    def generate_resume(self, profile_type: str = "random") -> Dict[str, Any]:
        """Generate a single resume with specified profile type"""
        
        # Generate basic info
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        name = f"{first_name} {last_name}"
        email = f"{first_name.lower()}.{last_name.lower()}@email.com"
        phone = fake.phone_number()
        location = random.choice(self.job_position_generator.locations)
        
        # Determine experience level based on profile type
        if profile_type == "junior":
            total_exp_years = random.uniform(0, 3)
            num_positions = random.randint(1, 2)
        elif profile_type == "mid":
            total_exp_years = random.uniform(3, 7)
            num_positions = random.randint(2, 4)
        elif profile_type == "senior":
            total_exp_years = random.uniform(7, 15)
            num_positions = random.randint(3, 5)
        else:  # random
            total_exp_years = random.uniform(0, 15)
            num_positions = random.randint(1, 5)
        
        # Generate education
        education = []
        degree_level = random.choice(["bachelors", "masters", "phd"]) if total_exp_years > 5 else "bachelors"
        
        education.append({
            "degree": f"{degree_level.title()} in {random.choice(self.degrees)}",
            "field": random.choice(self.degrees),
            "institution": random.choice(self.universities),
            "graduation_year": 2024 - int(total_exp_years) - 4,
            "level": degree_level
        })
        
        # Generate experience
        experience = []
        remaining_years = total_exp_years
        
        for i in range(num_positions):
            if i == 0:  # Current position
                duration_months = min(int(remaining_years * 12), random.randint(12, 48))
                end_date = None
                is_current = True
            else:
                duration_months = min(int(remaining_years * 12), random.randint(6, 36))
                end_date = datetime.now() - timedelta(days=random.randint(30, 365))
                is_current = False
            
            start_date = datetime.now() - timedelta(days=duration_months * 30)
            
            # Generate position details
            company = random.choice(self.job_position_generator.companies)
            position_title = self._generate_position_title(total_exp_years, i)
            
            exp_entry = {
                "company": company,
                "position": position_title,
                "duration_months": duration_months,
                "start_date": start_date.isoformat() if not is_current else None,
                "end_date": end_date.isoformat() if end_date else None,
                "description": self._generate_experience_description(position_title),
                "technologies": random.sample(self.job_position_generator.skills_pool["technical"], 
                                           random.randint(3, 8))
            }
            
            experience.append(exp_entry)
            remaining_years -= duration_months / 12
            
            if remaining_years <= 0:
                break
        
        # Generate skills based on experience
        all_skills = []
        for exp in experience:
            all_skills.extend(exp["technologies"])
        
        # Add additional skills
        additional_skills = random.sample(
            [s for s in self.job_position_generator.skills_pool["technical"] if s not in all_skills],
            min(random.randint(5, 10), len(self.job_position_generator.skills_pool["technical"]) - len(all_skills))
        )
        all_skills.extend(additional_skills)
        
        # Create skill objects
        skills = []
        for skill in list(set(all_skills)):
            skills.append({
                "name": skill,
                "proficiency": random.choice(["Beginner", "Intermediate", "Advanced", "Expert"]),
                "years_of_experience": random.uniform(1, min(total_exp_years, 10))
            })
        
        # Generate certifications
        certifications = []
        if random.random() > 0.5:
            cert_options = [
                "AWS Certified Solutions Architect",
                "Google Cloud Professional",
                "Certified Kubernetes Administrator",
                "PMP Certification",
                "Scrum Master Certification"
            ]
            certifications = random.sample(cert_options, random.randint(1, 3))
        
        # Generate summary
        summary = f"""Experienced {position_title} with {total_exp_years:.1f} years of expertise in 
        developing scalable solutions and leading technical initiatives. Proficient in 
        {', '.join(random.sample([s['name'] for s in skills], min(5, len(skills))))}. 
        Proven track record of delivering high-impact projects and mentoring teams."""
        
        # Generate salary expectations
        base_salary = 80000 + (total_exp_years * 10000)
        current_salary = base_salary + random.randint(-20000, 20000)
        expected_salary = current_salary + random.randint(10000, 40000)
        
        resume = {
            "id": f"resume_{str(uuid.uuid4())[:8]}",
            "name": name,
            "email": email,
            "phone": phone,
            "location": location,
            "summary": summary.replace('\n', ' ').strip(),
            "education": education,
            "experience": experience,
            "skills": skills,
            "certifications": certifications,
            "languages": ["English"] + (["Spanish"] if random.random() > 0.7 else []),
            "total_experience_years": round(total_exp_years, 1),
            "current_salary": current_salary,
            "expected_salary": expected_salary,
            "raw_text": self._generate_raw_text(name, email, phone, location, summary, 
                                               education, experience, skills, certifications),
            "parsed_at": datetime.now().isoformat()
        }
        
        return resume
    
    def _generate_position_title(self, total_exp: float, position_index: int) -> str:
        """Generate appropriate position title based on experience"""
        if total_exp < 3:
            titles = ["Junior Software Engineer", "Software Developer", "Engineer I"]
        elif total_exp < 7:
            titles = ["Software Engineer", "Senior Software Engineer", "Engineer II"]
        else:
            titles = ["Senior Software Engineer", "Staff Engineer", "Principal Engineer", 
                     "Technical Lead", "Engineering Manager"]
        
        # Progress through titles based on position index (0 = current)
        if position_index == 0:
            return titles[-1]  # Most senior for current position
        else:
            return titles[max(0, len(titles) - position_index - 1)]
    
    def _generate_experience_description(self, position: str) -> str:
        """Generate job description based on position"""
        descriptions = [
            "Developed and maintained scalable microservices architecture",
            "Led cross-functional team in delivering critical features",
            "Improved system performance by optimizing database queries",
            "Implemented CI/CD pipelines and automated testing",
            "Mentored junior developers and conducted code reviews",
            "Designed RESTful APIs and integrated third-party services",
            "Collaborated with product team to define technical requirements"
        ]
        return ". ".join(random.sample(descriptions, random.randint(3, 5)))
    
    def _generate_raw_text(self, name, email, phone, location, summary, 
                          education, experience, skills, certifications) -> str:
        """Generate raw text representation of resume"""
        text = f"{name}\n{email} | {phone} | {location}\n\n"
        text += f"SUMMARY\n{summary}\n\n"
        
        text += "EXPERIENCE\n"
        for exp in experience:
            text += f"{exp['position']} at {exp['company']}\n"
            text += f"{exp['description']}\n"
            text += f"Technologies: {', '.join(exp['technologies'])}\n\n"
        
        text += "EDUCATION\n"
        for edu in education:
            text += f"{edu['degree']} - {edu['institution']} ({edu['graduation_year']})\n\n"
        
        text += "SKILLS\n"
        text += ", ".join([s['name'] for s in skills]) + "\n\n"
        
        if certifications:
            text += "CERTIFICATIONS\n"
            text += "\n".join(certifications) + "\n"
        
        return text
    
    def generate_resumes(self, count: int = 50, 
                        distribution: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Generate multiple resumes with specified distribution"""
        if distribution is None:
            distribution = {
                "junior": 0.3,
                "mid": 0.5,
                "senior": 0.2
            }
        
        resumes = []
        
        for i in range(count):
            # Determine profile type based on distribution
            rand = random.random()
            if rand < distribution["junior"]:
                profile_type = "junior"
            elif rand < distribution["junior"] + distribution["mid"]:
                profile_type = "mid"
            else:
                profile_type = "senior"
            
            resume = self.generate_resume(profile_type)
            resumes.append(resume)
        
        return resumes
    
    def save_resumes(self, resumes: List[Dict[str, Any]], filename: str = "resumes_50.json"):
        """Save resumes to JSON file"""
        filepath = os.path.join("tests/data", filename)
        with open(filepath, 'w') as f:
            json.dump(resumes, f, indent=2)
        print(f"✅ Saved {len(resumes)} resumes to {filepath}")
        return filepath


def generate_test_data():
    """Generate complete test dataset"""
    print("🎯 Generating test data for 300 positions and 50 resumes...")
    
    # Generate job positions
    job_generator = JobPositionGenerator()
    positions = job_generator.generate_positions(300)
    positions_file = job_generator.save_positions(positions)
    
    # Generate resumes
    resume_generator = ResumeGenerator()
    resumes = resume_generator.generate_resumes(50)
    resumes_file = resume_generator.save_resumes(resumes)
    
    # Generate some perfect matches (planted data for validation)
    perfect_matches = []
    for i in range(5):
        position = random.choice(positions)
        
        # Create a resume that perfectly matches this position
        perfect_resume = resume_generator.generate_resume("senior")
        perfect_resume["skills"] = [
            {"name": skill, "proficiency": "Expert", "years_of_experience": 5.0}
            for skill in position["required_skills"]
        ]
        perfect_resume["total_experience_years"] = position["min_experience_years"] + 1
        perfect_resume["location"] = position["location"]
        perfect_resume["expected_salary"] = (position["salary_range_min"] + position["salary_range_max"]) / 2
        
        perfect_matches.append({
            "resume_id": perfect_resume["id"],
            "position_id": position["id"],
            "expected_score": 0.9  # Should score very high
        })
        
        resumes.append(perfect_resume)
    
    # Save updated resumes with perfect matches
    resume_generator.save_resumes(resumes, "resumes_with_perfect_matches.json")
    
    # Save perfect matches mapping
    with open("tests/data/perfect_matches.json", 'w') as f:
        json.dump(perfect_matches, f, indent=2)
    
    print(f"✅ Generated {len(positions)} job positions")
    print(f"✅ Generated {len(resumes)} resumes (including {len(perfect_matches)} perfect matches)")
    print("✅ Test data generation complete!")
    
    return positions_file, resumes_file


if __name__ == "__main__":
    generate_test_data()
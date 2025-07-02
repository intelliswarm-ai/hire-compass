from typing import Dict, Any, List
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.tools import Tool
from agents.base_agent import BaseAgent
from models.schemas import AspirationProfile, WorkMode
import re
import logging

logger = logging.getLogger(__name__)

class AspirationAgent(BaseAgent):
    """Agent responsible for analyzing candidate career aspirations and matching with positions"""
    
    def __init__(self):
        super().__init__("Aspiration")
        self.tools = self.create_tools()
        self.prompt = self.create_prompt()
        self.agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    def create_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are a career counselor AI specializing in understanding career aspirations.
            
            Your task is to:
            1. Extract career goals and aspirations from resume text and conversations
            2. Identify preferred work environments and company cultures
            3. Understand growth areas and learning interests
            4. Recognize areas the candidate wants to avoid
            5. Match aspirations with position opportunities
            
            Analyze patterns in:
            - Career progression and transitions
            - Project choices and initiatives
            - Skills development trajectory
            - Industry movements
            - Leadership and responsibility growth
            
            Provide insights on alignment between candidate aspirations and positions.
            """),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
    
    def create_tools(self) -> List[Tool]:
        return [
            Tool(
                name="extract_career_trajectory",
                func=self._extract_career_trajectory,
                description="Extract career progression patterns from work history"
            ),
            Tool(
                name="identify_growth_interests",
                func=self._identify_growth_interests,
                description="Identify areas of professional growth interest"
            ),
            Tool(
                name="analyze_role_preferences",
                func=self._analyze_role_preferences,
                description="Analyze preferences for types of roles and responsibilities"
            ),
            Tool(
                name="calculate_aspiration_alignment",
                func=self._calculate_aspiration_alignment,
                description="Calculate alignment between aspirations and position"
            )
        ]
    
    def _extract_career_trajectory(self, experience_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract career progression patterns"""
        trajectory = {
            "progression_type": "",
            "industry_changes": 0,
            "role_evolution": [],
            "responsibility_growth": "",
            "patterns": []
        }
        
        if not experience_data:
            return trajectory
        
        # Sort by date (most recent first)
        sorted_exp = sorted(experience_data, 
                          key=lambda x: x.get('end_date', '9999'), 
                          reverse=True)
        
        # Analyze progression
        positions = [exp.get('position', '').lower() for exp in sorted_exp]
        
        # Check for management progression
        mgmt_keywords = ['manager', 'director', 'lead', 'head', 'chief', 'vp', 'president']
        has_mgmt_progression = any(any(kw in pos for kw in mgmt_keywords) 
                                 for pos in positions[:len(positions)//2])
        
        if has_mgmt_progression:
            trajectory["progression_type"] = "management_track"
            trajectory["patterns"].append("Moving towards leadership roles")
        
        # Check for technical progression
        tech_keywords = ['senior', 'principal', 'staff', 'architect', 'expert']
        has_tech_progression = any(any(kw in pos for kw in tech_keywords) 
                                 for pos in positions[:len(positions)//2])
        
        if has_tech_progression:
            trajectory["progression_type"] = "technical_track" if not has_mgmt_progression else "hybrid_track"
            trajectory["patterns"].append("Advancing in technical expertise")
        
        # Industry changes
        companies = [exp.get('company', '') for exp in sorted_exp]
        unique_industries = len(set(companies))
        trajectory["industry_changes"] = unique_industries - 1 if unique_industries > 0 else 0
        
        if trajectory["industry_changes"] > 2:
            trajectory["patterns"].append("Cross-industry experience seeker")
        
        # Role evolution
        trajectory["role_evolution"] = positions[:3]  # Last 3 roles
        
        return trajectory
    
    def _identify_growth_interests(self, resume_data: Dict[str, Any]) -> List[str]:
        """Identify areas where candidate shows growth interest"""
        interests = []
        
        # Analyze skills progression
        skills = resume_data.get('skills', [])
        recent_skills = [s for s in skills if isinstance(s, dict) and s.get('years_of_experience', 0) < 2]
        
        if recent_skills:
            interests.append(f"Recently learning: {', '.join([s['name'] for s in recent_skills[:3]])}")
        
        # Analyze certifications
        certs = resume_data.get('certifications', [])
        if certs:
            interests.append(f"Professional development through certifications")
        
        # Analyze project types
        experience = resume_data.get('experience', [])
        for exp in experience[:2]:  # Recent positions
            desc = exp.get('description', '').lower()
            
            if 'led' in desc or 'managed' in desc:
                interests.append("Leadership and team management")
            if 'innovate' in desc or 'implement' in desc or 'design' in desc:
                interests.append("Innovation and system design")
            if 'mentor' in desc or 'train' in desc:
                interests.append("Mentoring and knowledge sharing")
            if 'strategy' in desc or 'roadmap' in desc:
                interests.append("Strategic planning")
        
        return list(set(interests))  # Remove duplicates
    
    def _analyze_role_preferences(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze preferences for types of roles"""
        preferences = {
            "work_mode": None,
            "company_size": None,
            "role_type": None,
            "industry_preference": None
        }
        
        experience = resume_data.get('experience', [])
        
        if not experience:
            return preferences
        
        # Analyze company patterns
        companies = [exp.get('company', '') for exp in experience]
        
        # Detect startup vs corporate preference
        startup_keywords = ['startup', 'founded', 'co-founder', 'seed', 'series']
        corporate_keywords = ['corporation', 'inc.', 'ltd.', 'global', 'international']
        
        startup_count = sum(1 for c in companies if any(kw in c.lower() for kw in startup_keywords))
        corporate_count = sum(1 for c in companies if any(kw in c.lower() for kw in corporate_keywords))
        
        if startup_count > corporate_count:
            preferences["company_size"] = "startup"
        elif corporate_count > startup_count:
            preferences["company_size"] = "enterprise"
        else:
            preferences["company_size"] = "flexible"
        
        # Analyze role types
        positions = [exp.get('position', '').lower() for exp in experience]
        
        if any('consult' in p for p in positions):
            preferences["role_type"] = "consulting"
        elif any('research' in p for p in positions):
            preferences["role_type"] = "research"
        elif any('architect' in p or 'principal' in p for p in positions):
            preferences["role_type"] = "technical_leadership"
        else:
            preferences["role_type"] = "individual_contributor"
        
        return preferences
    
    def _calculate_aspiration_alignment(self, aspiration_profile: Dict[str, Any],
                                      position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate alignment between aspirations and position"""
        alignment = {
            "overall_score": 0.0,
            "aligned_aspects": [],
            "misaligned_aspects": [],
            "growth_opportunities": []
        }
        
        scores = []
        
        # Role alignment
        if aspiration_profile.get("desired_roles"):
            position_title = position_data.get("title", "").lower()
            role_match = any(role.lower() in position_title 
                           for role in aspiration_profile["desired_roles"])
            
            if role_match:
                scores.append(1.0)
                alignment["aligned_aspects"].append("Role matches career aspirations")
            else:
                scores.append(0.5)
                alignment["misaligned_aspects"].append("Role differs from stated preferences")
        
        # Work mode alignment
        if aspiration_profile.get("work_mode_preference"):
            if aspiration_profile["work_mode_preference"] == position_data.get("work_mode"):
                scores.append(1.0)
                alignment["aligned_aspects"].append("Work mode preference matches")
            else:
                scores.append(0.3)
                alignment["misaligned_aspects"].append("Work mode mismatch")
        
        # Location alignment
        if aspiration_profile.get("location_preferences"):
            position_location = position_data.get("location", "").lower()
            location_match = any(loc.lower() in position_location 
                               for loc in aspiration_profile["location_preferences"])
            
            if location_match or position_data.get("work_mode") == "remote":
                scores.append(1.0)
                alignment["aligned_aspects"].append("Location preference satisfied")
            else:
                scores.append(0.4)
                alignment["misaligned_aspects"].append("Location not in preferred list")
        
        # Growth opportunities
        position_desc = position_data.get("description", "").lower()
        growth_keywords = ['growth', 'learning', 'development', 'advancement', 'career']
        
        if any(kw in position_desc for kw in growth_keywords):
            alignment["growth_opportunities"].append("Position emphasizes career development")
            scores.append(0.8)
        
        # Industry alignment
        if aspiration_profile.get("desired_industries"):
            # This would need more sophisticated industry matching
            scores.append(0.7)  # Neutral score for now
        
        # Calculate overall score
        alignment["overall_score"] = sum(scores) / len(scores) if scores else 0.5
        
        return alignment
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process aspiration analysis request"""
        try:
            resume_data = input_data.get('resume_data')
            position_data = input_data.get('position_data')
            additional_context = input_data.get('context', '')
            
            if not resume_data:
                raise ValueError("resume_data is required")
            
            # Extract career trajectory
            trajectory = self._extract_career_trajectory(
                resume_data.get('experience', [])
            )
            
            # Identify growth interests
            growth_interests = self._identify_growth_interests(resume_data)
            
            # Analyze role preferences
            role_preferences = self._analyze_role_preferences(resume_data)
            
            # Create aspiration profile
            aspiration_profile = AspirationProfile(
                desired_roles=self._infer_desired_roles(trajectory, role_preferences),
                desired_industries=self._infer_industries(resume_data),
                preferred_company_size=role_preferences.get("company_size"),
                work_mode_preference=self._infer_work_mode_preference(resume_data),
                location_preferences=self._infer_location_preferences(resume_data),
                career_goals=self._synthesize_career_goals(trajectory, growth_interests),
                growth_areas=growth_interests,
                avoid_areas=[]  # Would need more context to determine
            )
            
            result = {
                "aspiration_profile": aspiration_profile.dict(),
                "career_trajectory": trajectory,
                "insights": self._generate_insights(trajectory, growth_interests, role_preferences)
            }
            
            # If position data provided, calculate alignment
            if position_data:
                alignment = self._calculate_aspiration_alignment(
                    aspiration_profile.dict(),
                    position_data
                )
                result["position_alignment"] = alignment
            
            logger.info(f"Completed aspiration analysis for {resume_data.get('name', 'candidate')}")
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Error in aspiration analysis: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _infer_desired_roles(self, trajectory: Dict[str, Any], 
                           preferences: Dict[str, Any]) -> List[str]:
        """Infer desired roles from trajectory and preferences"""
        roles = []
        
        if trajectory["progression_type"] == "management_track":
            roles.extend(["Engineering Manager", "Director", "VP of Engineering"])
        elif trajectory["progression_type"] == "technical_track":
            roles.extend(["Principal Engineer", "Staff Engineer", "Technical Architect"])
        elif trajectory["progression_type"] == "hybrid_track":
            roles.extend(["Technical Lead", "Engineering Manager", "Solution Architect"])
        
        if preferences["role_type"] == "consulting":
            roles.append("Technical Consultant")
        elif preferences["role_type"] == "research":
            roles.append("Research Engineer")
        
        return roles[:5]  # Top 5 roles
    
    def _infer_industries(self, resume_data: Dict[str, Any]) -> List[str]:
        """Infer preferred industries from experience"""
        industries = []
        companies = [exp.get('company', '') for exp in resume_data.get('experience', [])]
        
        # Simple keyword-based industry detection
        industry_keywords = {
            "technology": ["tech", "software", "saas", "cloud"],
            "finance": ["bank", "financial", "fintech", "capital"],
            "healthcare": ["health", "medical", "pharma", "bio"],
            "retail": ["retail", "commerce", "shop"],
            "consulting": ["consulting", "advisory", "solutions"]
        }
        
        for company in companies:
            company_lower = company.lower()
            for industry, keywords in industry_keywords.items():
                if any(kw in company_lower for kw in keywords):
                    industries.append(industry)
        
        return list(set(industries))[:3]  # Top 3 unique industries
    
    def _infer_work_mode_preference(self, resume_data: Dict[str, Any]) -> WorkMode:
        """Infer work mode preference from resume"""
        # Look for remote work indicators
        resume_text = resume_data.get('raw_text', '').lower()
        
        if 'remote' in resume_text:
            return WorkMode.REMOTE
        elif 'hybrid' in resume_text:
            return WorkMode.HYBRID
        else:
            return WorkMode.HYBRID  # Default to hybrid as most flexible
    
    def _infer_location_preferences(self, resume_data: Dict[str, Any]) -> List[str]:
        """Infer location preferences"""
        preferences = []
        
        # Current location
        current_location = resume_data.get('location')
        if current_location:
            preferences.append(current_location)
        
        # Previous locations from experience
        for exp in resume_data.get('experience', []):
            # Would need location data in experience entries
            pass
        
        return preferences[:3]  # Top 3 locations
    
    def _synthesize_career_goals(self, trajectory: Dict[str, Any], 
                               interests: List[str]) -> str:
        """Synthesize career goals from analysis"""
        goals = []
        
        if trajectory["progression_type"] == "management_track":
            goals.append("Progress into senior leadership positions")
        elif trajectory["progression_type"] == "technical_track":
            goals.append("Become a recognized technical expert and thought leader")
        
        if "Strategic planning" in interests:
            goals.append("Drive technical strategy and innovation")
        
        if "Mentoring and knowledge sharing" in interests:
            goals.append("Build and develop high-performing teams")
        
        return ". ".join(goals) if goals else "Continue professional growth and impact"
    
    def _generate_insights(self, trajectory: Dict[str, Any], 
                         interests: List[str], 
                         preferences: Dict[str, Any]) -> List[str]:
        """Generate insights about candidate aspirations"""
        insights = []
        
        # Career trajectory insights
        if trajectory["progression_type"]:
            insights.append(f"Following a {trajectory['progression_type'].replace('_', ' ')}")
        
        if trajectory["industry_changes"] > 2:
            insights.append("Values diverse industry experience")
        
        # Growth insights
        if len(interests) > 3:
            insights.append("Highly motivated learner with diverse interests")
        
        # Company preference insights
        if preferences["company_size"] == "startup":
            insights.append("Thrives in fast-paced, entrepreneurial environments")
        elif preferences["company_size"] == "enterprise":
            insights.append("Prefers structured, established organizations")
        
        # Role type insights
        if preferences["role_type"] == "technical_leadership":
            insights.append("Seeks positions combining technical depth with leadership")
        
        return insights
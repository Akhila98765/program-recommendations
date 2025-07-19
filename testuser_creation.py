# create_test_users.py
import requests
import json

# Your Flask app URL
BASE_URL = "http://localhost:5000"

# Test users data
test_users = [
    {
        "email": "priya.sharma@example.com",
        "password": "TestPass123!",
        "full_name": "Priya Sharma",
        "role": "Quality Engineer",
        "skill_level": "Intermediate",
        "interests": "Test automation and quality assurance for SAP applications",
        "preferred_skills": "Testing, Quality Assurance, Automation",
        "max_budget": 150000,
        "preferred_month": "November"
    },
    {
        "email": "arjun.gupta@example.com", 
        "password": "TestPass123!",
        "full_name": "Arjun Gupta",
        "role": "DevOps Engineer",
        "skill_level": "Advanced",
        "interests": "CI/CD automation and cloud infrastructure management",
        "preferred_skills": "DevOps, Automation, Cloud, Testing",
        "max_budget": 200000,
        "preferred_month": "December"
    },
    {
        "email": "ananya.singh@example.com",
        "password": "TestPass123!",
        "full_name": "Ananya Singh", 
        "role": "Quality Engineer",
        "skill_level": "Intermediate",
        "interests": "Test automation and quality assurance for SAP applications",
        "preferred_skills": "CBTA, Testing, Quality Assurance",
        "max_budget": 180000,
        "preferred_month": "December"
    },
    {
        "email": "rohit.verma@example.com",
        "password": "TestPass123!",
        "full_name": "Rohit Verma",
        "role": "Developer",
        "skill_level": "Intermediate", 
        "interests": "Full-stack development and test automation",
        "preferred_skills": "JavaScript, Testing, Automation",
        "max_budget": 120000,
        "preferred_month": "January"
    },
    {
        "email": "kavya.agarwal@example.com",
        "password": "TestPass123!",
        "full_name": "Kavya Agarwal",
        "role": "Analytics Engineer",
        "skill_level": "Advanced",
        "interests": "Data analytics and business intelligence", 
        "preferred_skills": "Analytics, Data Science, Machine Learning",
        "max_budget": 250000,
        "preferred_month": "February"
    }
]

def register_user(user_data):
    """Register a user and complete onboarding"""
    try:
        # Register user
        register_response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": user_data["email"],
            "password": user_data["password"],
            "full_name": user_data["full_name"]
        })
        
        if register_response.status_code == 200:
            print(f"✅ Registered {user_data['full_name']}")
            
            # Login to get session
            login_response = requests.post(f"{BASE_URL}/api/auth/login", json={
                "email": user_data["email"],
                "password": user_data["password"]
            })
            
            if login_response.status_code == 200:
                print(f"✅ Logged in {user_data['full_name']}")
                
                # Complete onboarding
                onboarding_response = requests.post(f"{BASE_URL}/api/onboarding", json={
                    "role": user_data["role"],
                    "skill_level": user_data["skill_level"],
                    "interests": user_data["interests"],
                    "preferred_skills": user_data["preferred_skills"],
                    "max_budget": user_data["max_budget"],
                    "preferred_month": user_data["preferred_month"]
                })
                
                if onboarding_response.status_code == 200:
                    print(f"✅ Completed onboarding for {user_data['full_name']}")
                    return True
                else:
                    print(f"❌ Onboarding failed for {user_data['full_name']}: {onboarding_response.text}")
            else:
                print(f"❌ Login failed for {user_data['full_name']}: {login_response.text}")
        else:
            print(f"❌ Registration failed for {user_data['full_name']}: {register_response.text}")
            
    except Exception as e:
        print(f"❌ Error registering {user_data['full_name']}: {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 Creating test users...")
    
    for user in test_users:
        register_user(user)
        print("---")
    
    print("✅ Test user creation complete!")
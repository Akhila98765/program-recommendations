import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def create_test_users():
    test_users = [
        {
            "email": "sarah.johnson@example.com",
            "password": "testpass123",
            "full_name": "Sarah Johnson",
            "role": "Quality Engineer",
            "skill_level": "Beginner",
            "interests": "Web application testing and manual testing",
            "preferred_skills": "Manual Testing, CBTA, Quality Assurance, Selenium",
            "max_budget": 150000,
            "preferred_month": "December"
        },
        {
            "email": "michael.chen@example.com", 
            "password": "testpass123",
            "full_name": "Michael Chen",
            "role": "Quality Engineer",
            "skill_level": "Intermediate",
            "interests": "Web application testing and automation",
            "preferred_skills": "CBTA, Quality Assurance, Selenium, API Testing",
            "max_budget": 180000,
            "preferred_month": "January"
        }
    ]
    
    created_users = []
    
    for user_data in test_users:
        try:
            # Create auth user
            response = supabase_client.auth.sign_up({
                "email": user_data["email"],
                "password": user_data["password"]
            })
            
            if response.user:
                user_id = response.user.id
                
                # Create/update profile
                profile_data = {
                    "id": user_id,
                    "email": user_data["email"],
                    "full_name": user_data["full_name"],
                    "role": user_data["role"],
                    "skill_level": user_data["skill_level"],
                    "interests": user_data["interests"],
                    "preferred_skills": user_data["preferred_skills"],
                    "max_budget": user_data["max_budget"],
                    "preferred_month": user_data["preferred_month"]
                }
                
                # Insert profile
                supabase_client.table("user_profiles").upsert(profile_data).execute()
                
                created_users.append({
                    "user_id": user_id,
                    "email": user_data["email"],
                    "full_name": user_data["full_name"]
                })
                
                print(f"✅ Created user: {user_data['full_name']} (ID: {user_id})")
                
        except Exception as e:
            print(f"❌ Error creating user {user_data['full_name']}: {e}")
    
    return created_users

def create_test_registrations(users):
    registrations = [
        # User 1 registrations
        {
            "email": "sarah.johnson@example.com",
            "programs": [
                ("PROG001", "SAP Testing Fundamentals"),
                ("PROG017", "Quality Assurance Fundamentals"),
                ("PROG013", "Manual Testing Best Practices"),
                ("WEB001", "Web Application Testing Bootcamp")
            ]
        },
        # User 2 registrations
        {
            "email": "michael.chen@example.com",
            "programs": [
                ("PROG001", "SAP Testing Fundamentals"),
                ("PROG002", "Advanced Test Automation with Selenium"),
                ("PROG017", "Quality Assurance Fundamentals"),
                ("PROG003", "API Testing Masterclass"),
                ("WEB002", "Web Automation Testing")
            ]
        }
    ]
    
    for reg_data in registrations:
        user = next((u for u in users if u["email"] == reg_data["email"]), None)
        if not user:
            continue
            
        for program_id, program_title in reg_data["programs"]:
            try:
                supabase_client.table("program_registrations").insert({
                    "user_id": user["user_id"],
                    "program_id": program_id,
                    "program_title": program_title
                }).execute()
                
                print(f"✅ Registered {user['full_name']} for {program_title}")
                
            except Exception as e:
                print(f"❌ Error registering {user['full_name']} for {program_title}: {e}")

if __name__ == "__main__":
    print("Creating test users...")
    users = create_test_users()
    
    print("\nCreating test registrations...")
    create_test_registrations(users)
    
    print("\n✅ Test data creation complete!")
import os
from supabase import create_client, Client
from dotenv import load_dotenv
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_user_profile(user_id: str):
    """Get user profile from Supabase"""
    try:
        print(f"Fetching profile for user_id: {user_id}")  # Debug
        response = supabase_client.table("user_profiles").select("*").eq("id", user_id).execute()
        print(f"Response data: {response.data}")  # Debug
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        print(f"Error fetching user profile: {e}")
        return None

def create_user_profile(user_id: str, email: str, full_name: str):
    """Create new user profile"""
    try:
        # First check if profile already exists
        existing = get_user_profile(user_id)
        if existing:
            print(f"Profile already exists for user {user_id}")
            return existing
        
        print(f"Creating profile for user_id: {user_id}")  # Debug
        response = supabase_client.table("user_profiles").insert({
            "id": user_id,
            "email": email,
            "full_name": full_name
        }).execute()
        
        print(f"Profile created successfully: {response.data}")  # Debug
        return response.data
    except Exception as e:
        print(f"Error creating user profile: {e}")
        # Try to create with upsert to handle race conditions
        try:
            response = supabase_client.table("user_profiles").upsert({
                "id": user_id,
                "email": email,
                "full_name": full_name
            }).execute()
            print(f"Profile upserted successfully: {response.data}")
            return response.data
        except Exception as e2:
            print(f"Error upserting user profile: {e2}")
            return None

def update_user_profile(user_id: str, updates: dict):
    """Update user profile"""
    try:
        print(f"Updating profile for user_id: {user_id} with data: {updates}")  # Debug
        response = supabase_client.table("user_profiles").update(updates).eq("id", user_id).execute()
        print(f"Update response: {response.data}")  # Debug
        return response.data
    except Exception as e:
        print(f"Error updating user profile: {e}")
        return None

def register_for_program(user_id: str, program_id: str, program_title: str):
    """Register user for a program"""
    try:
        # Check if already registered
        existing = supabase_client.table("program_registrations").select("*").eq("user_id", user_id).eq("program_id", program_id).execute()
        if existing.data:
            return {"success": False, "message": "Already registered for this program"}
        
        response = supabase_client.table("program_registrations").insert({
            "user_id": user_id,
            "program_id": program_id,
            "program_title": program_title
        }).execute()
        
        return {"success": True, "data": response.data}
    except Exception as e:
        print(f"Error registering for program: {e}")
        return {"success": False, "message": str(e)}

def get_user_registrations(user_id: str):
    """Get all programs user has registered for"""
    try:
        response = supabase_client.table("program_registrations").select("*").eq("user_id", user_id).execute()
        return response.data
    except Exception as e:
        print(f"Error fetching user registrations: {e}")
        return []

def unregister_from_program(user_id: str, program_id: str):
    """Unregister user from a program"""
    try:
        response = supabase_client.table("program_registrations").delete().eq("user_id", user_id).eq("program_id", program_id).execute()
        return {"success": True, "data": response.data}
    except Exception as e:
        print(f"Error unregistering from program: {e}")
        return {"success": False, "message": str(e)}

def get_all_registrations():
    """Get all program registrations for collaborative filtering"""
    try:
        response = supabase_client.table("program_registrations").select("user_id, program_id, program_title").execute()
        return response.data
    except Exception as e:
        print(f"Error fetching all registrations: {e}")
        return []

def get_users_with_similar_profiles(current_user_id: str, limit: int = 10):
    """Get users with similar profiles (role, skill_level, interests)"""
    try:
        print(f"\n=== FINDING SIMILAR USERS FOR: {current_user_id} ===")
        
        # Get current user's profile
        current_profile = get_user_profile(current_user_id)
        if not current_profile:
            print("❌ Current user profile not found!")
            return []
        
        print(f"📋 Current user profile:")
        print(f"   Role: {current_profile.get('role', 'N/A')}")
        print(f"   Skill Level: {current_profile.get('skill_level', 'N/A')}")
        print(f"   Interests: {current_profile.get('interests', 'N/A')}")
        
        # Get all user profiles
        all_profiles = supabase_client.table("user_profiles").select("*").execute()
        print(f"📊 Total profiles found: {len(all_profiles.data)}")
        
        similar_users = []
        current_role = current_profile.get('role', '').lower()
        current_skill = current_profile.get('skill_level', '').lower()
        current_interests = current_profile.get('interests', '').lower()
        
        for profile in all_profiles.data:
            if profile['id'] == current_user_id:
                continue
                
            # Calculate similarity score
            similarity_score = 0
            
            # Role similarity
            if profile.get('role', '').lower() == current_role:
                similarity_score += 3
                print(f"   👥 Role match with {profile.get('full_name', 'Unknown')}: +3")
            
            # Skill level similarity
            if profile.get('skill_level', '').lower() == current_skill:
                similarity_score += 2
                print(f"   📈 Skill level match with {profile.get('full_name', 'Unknown')}: +2")
            
            # Interest similarity (check if any words match)
            profile_interests = profile.get('interests', '').lower()
            if profile_interests and current_interests:
                current_words = set(current_interests.split())
                profile_words = set(profile_interests.split())
                common_words = current_words.intersection(profile_words)
                if common_words:
                    similarity_score += len(common_words)
                    print(f"   🎯 Interest match with {profile.get('full_name', 'Unknown')}: +{len(common_words)} (words: {common_words})")
            
            if similarity_score > 0:
                similar_users.append({
                    'user_id': profile['id'],
                    'similarity_score': similarity_score,
                    'profile': profile
                })
                print(f"   ✅ Added similar user: {profile.get('full_name', 'Unknown')} (score: {similarity_score})")
        
        # Sort by similarity score
        similar_users.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        print(f"\n🔍 Found {len(similar_users)} similar users")
        for i, user in enumerate(similar_users[:limit]):
            print(f"   {i+1}. {user['profile'].get('full_name', 'Unknown')} (score: {user['similarity_score']})")
        
        return similar_users[:limit]
        
    except Exception as e:
        print(f"❌ Error finding similar users: {e}")
        return []

def get_collaborative_recommendations(user_id: str, limit: int = 10):
    """Get program recommendations based on collaborative filtering"""
    try:
        print(f"\n=== COLLABORATIVE RECOMMENDATIONS FOR: {user_id} ===")
        
        # Get current user's registrations first
        current_registrations = get_user_registrations(user_id)
        current_program_ids = {reg['program_id'] for reg in current_registrations}
        
        print(f"📚 Current user is registered for {len(current_program_ids)} programs:")
        for reg in current_registrations:
            print(f"   - {reg['program_title']} (ID: {reg['program_id']})")
        
        # Get similar users
        similar_users = get_users_with_similar_profiles(user_id, limit=20)
        if not similar_users:
            print("❌ No similar users found!")
            return []
        
        print(f"✅ Found {len(similar_users)} similar users")
        
        # Get programs registered by similar users
        program_scores = defaultdict(int)
        program_titles = {}
        
        for similar_user in similar_users:
            user_registrations = get_user_registrations(similar_user['user_id'])
            similarity_weight = similar_user['similarity_score']
            
            print(f"\n👤 Analyzing {similar_user['profile'].get('full_name', 'Unknown')} (weight: {similarity_weight})")
            print(f"   Registered for {len(user_registrations)} programs:")
            
            for reg in user_registrations:
                program_id = reg['program_id']
                program_title = reg['program_title']
                
                print(f"   - {program_title} (ID: {program_id})")
                
                # Don't recommend programs user has already registered for
                if program_id not in current_program_ids:
                    program_scores[program_id] += similarity_weight
                    program_titles[program_id] = program_title
                    print(f"     ✅ Added to recommendations (current score: {program_scores[program_id]})")
                else:
                    print(f"     ⚠️  Already registered - skipping")
        
        print(f"\n📊 COLLABORATIVE PROGRAM SCORES:")
        for program_id, score in sorted(program_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   {program_titles[program_id]} (ID: {program_id}): {score}")
        
        # Sort by score and return top recommendations
        sorted_programs = sorted(program_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for program_id, score in sorted_programs[:limit]:
            recommendations.append({
                'program_id': program_id,
                'program_title': program_titles[program_id],
                'collaborative_score': score,
                'recommendation_type': 'collaborative'
            })
        
        print(f"\n🎯 TOP {len(recommendations)} COLLABORATIVE RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations):
            print(f"   {i+1}. {rec['program_title']} (score: {rec['collaborative_score']})")
        
        return recommendations
        
    except Exception as e:
        print(f"❌ Error getting collaborative recommendations: {e}")
        return []

def get_program_registration_count(program_id: str):
    """Get count of users registered for a specific program"""
    try:
        response = supabase_client.table("program_registrations").select("user_id").eq("program_id", program_id).execute()
        count = len(response.data)
        print(f"📊 Program {program_id} has {count} registrations")
        return count
    except Exception as e:
        print(f"Error getting registration count: {e}")
        return 0
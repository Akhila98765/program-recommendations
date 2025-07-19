from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from pinecone_utils import search_similar_programs
from supabase_utils import (
    supabase_client, get_user_profile, create_user_profile, update_user_profile,
    register_for_program, get_user_registrations, unregister_from_program,
    get_collaborative_recommendations, get_program_registration_count
)
from datetime import datetime
import calendar
import jwt
import os
from functools import wraps

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'Madan')

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = session.get('access_token')
        if not token:
            return redirect(url_for('login'))
        
        try:
            # Verify token with Supabase
            user = supabase_client.auth.get_user(token)
            if not user:
                return redirect(url_for('login'))
            session['user_id'] = user.user.id
        except:
            return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated

def month_to_number(month_name):
    """Convert month name to number"""
    try:
        return list(calendar.month_name).index(month_name.capitalize())
    except ValueError:
        try:
            return list(calendar.month_abbr).index(month_name.capitalize())
        except ValueError:
            return None

def is_program_available_in_month(start_date, end_date, target_month_num):
    """Check if program runs during the target month"""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Check if target month falls within the program duration
        return start.month <= target_month_num <= end.month or \
               (start.year < end.year and (start.month <= target_month_num or target_month_num <= end.month))
    except:
        return True

def generate_content_based_recommendations(profile, user_registrations):
    """Generate content-based recommendations using vector search"""
    if not profile:
        return []
    
    # Get list of registered program IDs
    registered_program_ids = {reg['program_id'] for reg in user_registrations}
    
    # Use profile data for recommendations
    interest = profile.get('interests', '')
    role = profile.get('role', '')
    skill_level = profile.get('skill_level', '')
    skills = profile.get('preferred_skills', '')
    available_month = profile.get('preferred_month', '')
    max_cost = profile.get('max_budget', 0)
    
    # Convert month name to number for filtering
    target_month_num = month_to_number(available_month)
    if not target_month_num:
        return []
    
    # Build Pinecone filters
    filters = {}
    if max_cost and max_cost > 0:
        filters["cost"] = {"$lte": float(max_cost)}
    
    # Create composite query
    full_query = f"{interest}. Role: {role}. Skills to learn: {skills}. Level: {skill_level}. Available in: {available_month}."
    
    try:
        # Get only top 10 from Pinecone
        results = search_similar_programs(full_query, filters=filters, top_k=10)
        
        # Post-process results
        recommendations = []
        
        for match in results:
            metadata = match.metadata
            program_id = metadata.get("program_id")
            
            # Skip if user is already registered for this program
            if program_id in registered_program_ids:
                continue
            
            # Only include high-quality matches (score > 0.7)
            if match.score > 0.7 and is_program_available_in_month(
                metadata.get("start_date"), 
                metadata.get("end_date"), 
                target_month_num
            ):
                recommendations.append({
                    "program_id": program_id,
                    "title": metadata.get("title"),
                    "category": metadata.get("category"),
                    "skills_required": metadata.get("skills_required"),
                    "cost": metadata.get("cost"),
                    "start_date": metadata.get("start_date"),
                    "end_date": metadata.get("end_date"),
                    "score": round(match.score, 3),
                    "recommendation_type": "content_based"
                })
        
        # Sort by relevance score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top 3 content-based recommendations (to leave room for collaborative)
        return recommendations[:3]
    
    except Exception as e:
        print(f"Error generating content-based recommendations: {e}")
        return []

def merge_recommendations(content_based, collaborative, user_registrations):
    """Merge content-based and collaborative recommendations"""
    print(f"\n=== MERGING RECOMMENDATIONS ===")
    print(f"📋 Content-based recommendations: {len(content_based)}")
    print(f"👥 Collaborative recommendations: {len(collaborative)}")
    print(f"📚 User registrations: {len(user_registrations)}")
    
    registered_program_ids = {reg['program_id'] for reg in user_registrations}
    
    # Process content-based recommendations (already filtered)
    for rec in content_based:
        rec['is_registered'] = False  # Already filtered out
        rec['collaborative_info'] = None
    
    print(f"✅ Processed {len(content_based)} content-based recommendations")
    
    # Process collaborative recommendations and get their full details
    collaborative_with_details = []
    for collab_rec in collaborative:
        program_id = collab_rec['program_id']
        
        # Skip if user is already registered for this program
        if program_id in registered_program_ids:
            print(f"⚠️  Skipping {collab_rec['program_title']} - user already registered")
            continue
        
        # Skip if already in content-based recommendations
        if any(cb['program_id'] == program_id for cb in content_based):
            print(f"⚠️  Skipping {collab_rec['program_title']} - already in content-based")
            continue
            
        # Get program details from Pinecone (or you could store in Supabase)
        try:
            # For now, create a basic structure - in production, you'd fetch full details
            collaborative_with_details.append({
                "program_id": program_id,
                "title": collab_rec['program_title'],
                "category": "Collaborative",
                "skills_required": "Based on similar users",
                "cost": 0,  # You'd fetch actual cost from your data
                "start_date": "TBD",
                "end_date": "TBD",
                "score": collab_rec['collaborative_score'] / 10,  # Normalize score
                "recommendation_type": "collaborative",
                "is_registered": False,  # Already filtered out
                "collaborative_info": {
                    "users_registered": get_program_registration_count(program_id),
                    "message": "Users similar to you have registered for this program"
                }
            })
            print(f"✅ Added collaborative recommendation: {collab_rec['program_title']}")
        except Exception as e:
            print(f"❌ Error processing collaborative recommendation: {e}")
            continue
    
    print(f"✅ Processed {len(collaborative_with_details)} collaborative recommendations")
    
    # Combine recommendations
    all_recommendations = content_based + collaborative_with_details
    
    # Sort by score (prioritize content-based with higher scores, then collaborative)
    all_recommendations.sort(key=lambda x: (x['recommendation_type'] == 'collaborative', -x['score']))
    
    print(f"🎯 FINAL MERGED RECOMMENDATIONS ({len(all_recommendations)} total):")
    for i, rec in enumerate(all_recommendations):
        print(f"   {i+1}. {rec['title']} ({rec['recommendation_type']}) - Score: {rec['score']}")
    
    # Return only top 5 recommendations
    return all_recommendations[:5]

@app.route("/")
def home():
    if 'access_token' not in session:
        return redirect(url_for('login'))
    return render_template("dashboard.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/onboarding")
@token_required
def onboarding():
    return render_template("onboarding.html")

@app.route("/profile")
@token_required
def profile():
    return render_template("profile.html")

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    try:
        response = supabase_client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        session['access_token'] = response.session.access_token
        session['user_id'] = response.user.id
        
        # Check if user has completed onboarding
        profile = get_user_profile(response.user.id)
        
        if not profile or not profile.get('role'):
            return jsonify({"success": True, "redirect": "/onboarding"})
        
        return jsonify({"success": True, "redirect": "/"})
    
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('full_name')
    
    try:
        response = supabase_client.auth.sign_up({
            "email": email,
            "password": password
        })
        
        # Set session immediately after successful registration
        if response.session:
            session['access_token'] = response.session.access_token
            session['user_id'] = response.user.id
            
            # Create user profile with authenticated context
            create_user_profile(response.user.id, email, full_name)
            
            return jsonify({"success": True, "redirect": "/onboarding"})
        else:
            # Email confirmation required
            return jsonify({"success": True, "message": "Please check your email for confirmation", "redirect": "/login"})
    
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/api/onboarding", methods=["POST"])
@token_required
def complete_onboarding():
    data = request.json
    user_id = session['user_id']
    
    try:
        # First, ensure the user profile exists
        profile = get_user_profile(user_id)
        if not profile:
            # Create profile if it doesn't exist (fallback)
            user = supabase_client.auth.get_user(session['access_token'])
            create_user_profile(user_id, user.user.email, user.user.user_metadata.get('full_name', ''))
        
        # Update profile with onboarding data
        update_user_profile(user_id, {
            'role': data.get('role'),
            'skill_level': data.get('skill_level'),
            'interests': data.get('interests'),
            'preferred_skills': data.get('preferred_skills'),
            'max_budget': data.get('max_budget'),
            'preferred_month': data.get('preferred_month')
        })
        
        return jsonify({"success": True})
    
    except Exception as e:
        print(f"Onboarding error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/profile", methods=["GET"])
@token_required
def get_profile():
    user_id = session['user_id']
    profile = get_user_profile(user_id)
    
    if not profile:
        return jsonify({"error": "Profile not found"}), 404
    
    return jsonify(profile)

@app.route("/api/profile", methods=["PUT"])
@token_required
def update_profile():
    data = request.json
    user_id = session['user_id']
    
    try:
        # Update profile
        update_user_profile(user_id, {
            'full_name': data.get('full_name'),
            'role': data.get('role'),
            'skill_level': data.get('skill_level'),
            'interests': data.get('interests'),
            'preferred_skills': data.get('preferred_skills'),
            'max_budget': data.get('max_budget'),
            'preferred_month': data.get('preferred_month')
        })
        
        return jsonify({"success": True})
    
    except Exception as e:
        print(f"Profile update error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/recommendations", methods=["GET"])
@token_required
def get_recommendations():
    user_id = session['user_id']
    print(f"\n🚀 GETTING RECOMMENDATIONS FOR USER: {user_id}")
    
    profile = get_user_profile(user_id)
    
    if not profile:
        print("❌ Profile not found!")
        return jsonify({"error": "Profile not found"}), 404
    
    print(f"✅ User profile loaded: {profile.get('full_name', 'Unknown')}")
    
    # Get user registrations FIRST
    print("\n📚 Getting user registrations...")
    user_registrations = get_user_registrations(user_id)
    print(f"✅ User registrations: {len(user_registrations)}")
    
    # Get content-based recommendations (filtered)
    print("\n🔍 Getting content-based recommendations...")
    content_based = generate_content_based_recommendations(profile, user_registrations)
    print(f"✅ Content-based recommendations: {len(content_based)}")
    
    # Get collaborative recommendations - reduced limit
    print("\n👥 Getting collaborative recommendations...")
    collaborative = get_collaborative_recommendations(user_id, limit=5)
    print(f"✅ Collaborative recommendations: {len(collaborative)}")
    
    # Merge recommendations
    print("\n🔄 Merging recommendations...")
    merged_recommendations = merge_recommendations(content_based, collaborative, user_registrations)
    print(f"✅ Final recommendations: {len(merged_recommendations)}")
    
    # Return only top 5 most apt recommendations
    return jsonify({"recommendations": merged_recommendations[:5]})

@app.route("/api/register-program", methods=["POST"])
@token_required
def register_program():
    data = request.json
    user_id = session['user_id']
    program_id = data.get('program_id')
    program_title = data.get('program_title')
    
    if not program_id or not program_title:
        return jsonify({"success": False, "error": "Missing program details"}), 400
    
    result = register_for_program(user_id, program_id, program_title)
    
    if result['success']:
        return jsonify({"success": True, "message": "Successfully registered for program"})
    else:
        return jsonify({"success": False, "error": result['message']}), 400

@app.route("/api/unregister-program", methods=["POST"])
@token_required
def unregister_program():
    data = request.json
    user_id = session['user_id']
    program_id = data.get('program_id')
    
    if not program_id:
        return jsonify({"success": False, "error": "Missing program ID"}), 400
    
    result = unregister_from_program(user_id, program_id)
    
    if result['success']:
        return jsonify({"success": True, "message": "Successfully unregistered from program"})
    else:
        return jsonify({"success": False, "error": result['message']}), 400

@app.route("/api/my-registrations", methods=["GET"])
@token_required
def get_my_registrations():
    user_id = session['user_id']
    registrations = get_user_registrations(user_id)
    return jsonify({"registrations": registrations})

@app.route("/recommend", methods=["POST"])
@token_required
def recommend_programs():
    data = request.json
    user_id = session['user_id']
    
    interest = data.get("interest")
    role = data.get("role")
    skill_level = data.get("skill_level")
    skills = data.get("skills")
    available_month = data.get("available_month")
    max_cost = data.get("max_cost")
    
    if not all([interest, role, skill_level, skills, available_month]):
        return jsonify({"error": "Missing required fields"}), 400
    
    # Get user registrations first
    registrations = get_user_registrations(user_id)
    registered_program_ids = {reg['program_id'] for reg in registrations}
    
    # Convert month name to number
    target_month_num = month_to_number(available_month)
    if not target_month_num:
        return jsonify({"error": "Invalid month format"}), 400
    
    # Build filters
    filters = {}
    if max_cost and max_cost > 0:
        filters["cost"] = {"$lte": float(max_cost)}
    
    # Create query
    full_query = f"{interest}. Role: {role}. Skills to learn: {skills}. Level: {skill_level}. Available in: {available_month}."
    
    try:
        # Get only top 10 from Pinecone
        results = search_similar_programs(full_query, filters=filters, top_k=10)
        recommendations = []
        
        for match in results:
            metadata = match.metadata
            program_id = metadata.get("program_id")
            
            # Skip if user is already registered for this program
            if program_id in registered_program_ids:
                continue
            
            # Only include high-quality matches (score > 0.7)
            if match.score > 0.6 and is_program_available_in_month(
                metadata.get("start_date"), 
                metadata.get("end_date"), 
                target_month_num
            ):
                recommendations.append({
                    "program_id": program_id,
                    "title": metadata.get("title"),
                    "category": metadata.get("category"),
                    "skills_required": metadata.get("skills_required"),
                    "cost": metadata.get("cost"),
                    "start_date": metadata.get("start_date"),
                    "end_date": metadata.get("end_date"),
                    "score": round(match.score, 3),
                    "recommendation_type": "search",
                    "is_registered": False,  # Already filtered out
                    "collaborative_info": None
                })
        
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top 5 most apt search results
        return jsonify({"recommendations": recommendations[:5]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

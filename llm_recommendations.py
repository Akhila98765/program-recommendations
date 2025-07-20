import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from nomic import embed
from pinecone import Pinecone
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192",
    temperature=0.3
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("programs")

def embed_text_nomic(text: str, input_type: str = "search_query"):
    """Embed text using Nomic's embedding model"""
    try:
        output = embed.text(
            texts=[f"{input_type}: {text}"],
            model='nomic-embed-text-v1.5'
        )
        return output['embeddings'][0]
    except Exception as e:
        print(f"Error generating Nomic embedding: {e}")
        return None

class LLMRecommendationEngine:
    def __init__(self):
        self.llm = llm
        self.index = index
        
        # Main recommendation prompt - 3 recommendations
        self.recommendation_prompt = ChatPromptTemplate.from_template("""
You are an expert learning and development advisor. Your task is to recommend the most suitable learning programs for an employee based on their profile and available programs.

EMPLOYEE PROFILE:
- Name: {full_name}
- Role: {role}
- Skill Level: {skill_level}
- Interests: {interests}
- Preferred Skills: {preferred_skills}
- Budget: ${max_budget}
- Preferred Month: {preferred_month}

CURRENTLY REGISTERED PROGRAMS:
{user_registrations}

AVAILABLE PROGRAMS FROM SEARCH:
{search_results}

INSTRUCTIONS:
1. Analyze the employee's profile, current skill level, role, and learning interests
2. Consider their budget constraints and timing preferences
3. Avoid recommending programs they're already registered for
4. Prioritize programs that:
   - Align with their role and career progression
   - Match their skill level (not too basic, not too advanced)
   - Fit within their budget
   - Are relevant to their stated interests and preferred skills
5. Provide diversity in recommendations (different categories/skills)

RESPONSE FORMAT:
Return a JSON array of exactly 3 recommended programs with this structure:
[
  {{
    "program_id": "program_id_here",
    "title": "Program Title",
    "recommendation_score": 0.95,
    "recommendation_reason": "Detailed explanation why this program is perfect for this employee",
    "category": "Program Category",
    "skills_gained": "Key skills they will learn",
    "career_impact": "How this will help their career progression",
    "urgency": "high/medium/low - based on current industry trends and their role"
  }}
]

Only return valid JSON, no additional text.
""")
        
        # Enhancement prompt for other recommendation types
        self.enhancement_prompt = ChatPromptTemplate.from_template("""
You are a learning advisor. Enhance this program recommendation with detailed insights for the employee.

EMPLOYEE PROFILE:
- Role: {role}
- Skill Level: {skill_level}
- Interests: {interests}
- Preferred Skills: {preferred_skills}

PROGRAM TO ENHANCE:
- Title: {program_title}
- Category: {program_category}
- Skills Required: {program_skills}
- Recommendation Source: {recommendation_source}

INSTRUCTIONS:
Provide detailed enhancement for this {recommendation_source} recommendation:
1. Explain why this program fits the employee's profile
2. Detail specific skills they'll gain
3. Describe career impact
4. Assess urgency based on industry trends

RESPONSE FORMAT:
Return JSON with this structure:
{{
  "recommendation_reason": "Detailed explanation why this program fits their profile and goals",
  "skills_gained": "Specific skills and competencies they will develop",
  "career_impact": "How this will advance their career and open new opportunities",
  "urgency": "high/medium/low",
  "enhanced_explanation": "Additional context about why this {recommendation_source} recommendation is valuable"
}}

Only return valid JSON, no additional text.
""")
        
        # Create chains
        self.json_parser = JsonOutputParser()
        self.chain = self.recommendation_prompt | self.llm | self.json_parser
        self.enhancement_chain = self.enhancement_prompt | self.llm | self.json_parser

    def get_enhanced_search_results(self, user_profile, top_k=20):
        """Get comprehensive search results from Pinecone using Nomic embeddings"""
        
        # Build comprehensive query
        query_parts = []
        
        if user_profile.get('interests'):
            query_parts.append(f"Learning interests: {user_profile['interests']}")
        
        if user_profile.get('role'):
            query_parts.append(f"Job role: {user_profile['role']}")
            
        if user_profile.get('preferred_skills'):
            query_parts.append(f"Skills to develop: {user_profile['preferred_skills']}")
            
        if user_profile.get('skill_level'):
            query_parts.append(f"Current level: {user_profile['skill_level']}")
        
        query = ". ".join(query_parts)
        print(f"🔍 Enhanced search query: {query}")
        
        # Add filters
        filters = {}
        if user_profile.get('max_budget'):
            try:
                max_budget = float(user_profile['max_budget'])
                filters["cost"] = {"$lte": max_budget}
                print(f"💰 Budget filter: <= ${max_budget}")
            except (ValueError, TypeError):
                print("⚠️ Invalid budget value, skipping budget filter")
        
        # Generate embedding using Nomic
        try:
            query_embedding = embed_text_nomic(query, input_type="search_query")
            
            if query_embedding is None:
                print("❌ Failed to generate query embedding")
                return []
            
            print(f"✅ Query embedding generated, length: {len(query_embedding)}")
            
            # Search Pinecone
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filters or {}
            )
            
            print(f"📊 Pinecone returned {len(response.matches)} matches")
            
            # Format results
            formatted_results = []
            for i, match in enumerate(response.matches):
                metadata = match.metadata
                formatted_result = {
                    "program_id": metadata.get("program_id"),
                    "title": metadata.get("title"),
                    "category": metadata.get("category"),
                    "skills_required": metadata.get("skills_required"),
                    "cost": metadata.get("cost"),
                    "start_date": metadata.get("start_date"),
                    "end_date": metadata.get("end_date"),
                    "similarity_score": round(match.score, 3),
                    "description": metadata.get("description", ""),
                    "level": metadata.get("level", ""),
                    "duration": metadata.get("duration", "")
                }
                formatted_results.append(formatted_result)
                
                print(f"   {i+1}. {metadata.get('title', 'Unknown')} (Score: {match.score:.3f})")
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error in enhanced search: {e}")
            return []

    def enhance_recommendation_with_ai(self, user_profile, program_data, recommendation_source):
        """Use AI to enhance any recommendation with detailed insights"""
        try:
            print(f"🤖 Enhancing {recommendation_source} recommendation: {program_data.get('title', 'Unknown')}")
            
            # Prepare input for enhancement
            enhancement_input = {
                "role": user_profile.get('role', 'Professional'),
                "skill_level": user_profile.get('skill_level', 'Intermediate'),
                "interests": user_profile.get('interests', 'General'),
                "preferred_skills": user_profile.get('preferred_skills', 'Various'),
                "program_title": program_data.get('title', 'Program'),
                "program_category": program_data.get('category', 'Learning'),
                "program_skills": program_data.get('skills_required', 'Various skills'),
                "recommendation_source": recommendation_source
            }
            
            # Get AI enhancement
            enhancement = self.enhancement_chain.invoke(enhancement_input)
            
            print(f"✅ Enhanced {recommendation_source} recommendation with AI insights")
            return enhancement
            
        except Exception as e:
            print(f"❌ Error enhancing recommendation: {e}")
            # Fallback enhancement
            return {
                "recommendation_reason": f"This {recommendation_source} program aligns with your role and interests",
                "skills_gained": "Relevant professional skills",
                "career_impact": "Will contribute to your career advancement",
                "urgency": "medium",
                "enhanced_explanation": f"Selected through {recommendation_source} analysis"
            }

    def get_llm_recommendations(self, user_profile, user_registrations):
        """Get LLM-powered recommendations (3 items)"""
        try:
            print(f"\n🤖 GETTING LLM RECOMMENDATIONS FOR: {user_profile.get('full_name', 'Unknown')}")
            
            # Get comprehensive search results
            search_results = self.get_enhanced_search_results(user_profile, top_k=15)
            print(f"📊 Retrieved {len(search_results)} programs from vector search")
            
            if not search_results:
                print("❌ No search results found, cannot generate LLM recommendations")
                return []
            
            # Format user registrations
            registered_programs = []
            for reg in user_registrations:
                registered_programs.append(f"- {reg['program_title']} (ID: {reg['program_id']})")
            
            registrations_text = "\n".join(registered_programs) if registered_programs else "None"
            
            # Format search results for LLM
            search_text = json.dumps(search_results[:10], indent=1)
            
            # Prepare input variables for the chain
            chain_input = {
                "full_name": user_profile.get('full_name', 'Unknown'),
                "role": user_profile.get('role', 'Not specified'),
                "skill_level": user_profile.get('skill_level', 'Not specified'),
                "interests": user_profile.get('interests', 'Not specified'),
                "preferred_skills": user_profile.get('preferred_skills', 'Not specified'),
                "max_budget": user_profile.get('max_budget', 'Not specified'),
                "preferred_month": user_profile.get('preferred_month', 'Not specified'),
                "user_registrations": registrations_text,
                "search_results": search_text
            }
            
            # Generate recommendations using LLM
            print("🧠 Generating LLM recommendations...")
            try:
                recommendations = self.chain.invoke(chain_input)
                print(f"✅ LLM generated {len(recommendations)} recommendations")
                
                # Enhance recommendations with metadata
                enhanced_recommendations = []
                search_dict = {r['program_id']: r for r in search_results}
                
                for i, rec in enumerate(recommendations):
                    program_id = rec.get('program_id')
                    print(f"   Processing rec {i+1}: {program_id}")
                    
                    if program_id in search_dict:
                        search_data = search_dict[program_id]
                        
                        enhanced_rec = {
                            "program_id": program_id,
                            "title": rec.get('title', search_data['title']),
                            "category": rec.get('category', search_data['category']),
                            "skills_required": search_data.get('skills_required', ''),
                            "cost": search_data.get('cost', 0),
                            "start_date": search_data.get('start_date', ''),
                            "end_date": search_data.get('end_date', ''),
                            "score": rec.get('recommendation_score', 0.8),
                            "recommendation_type": "llm_powered",
                            "llm_reasoning": {
                                "reason": rec.get('recommendation_reason', ''),
                                "skills_gained": rec.get('skills_gained', ''),
                                "career_impact": rec.get('career_impact', ''),
                                "urgency": rec.get('urgency', 'medium')
                            },
                            "is_registered": False,
                            "similarity_score": search_data.get('similarity_score', 0)
                        }
                        enhanced_recommendations.append(enhanced_rec)
                        print(f"   ✅ Enhanced: {enhanced_rec['title']}")
                
                print(f"🎯 Enhanced {len(enhanced_recommendations)} LLM recommendations")
                return enhanced_recommendations
                
            except Exception as parse_error:
                print(f"❌ Error with JSON parsing: {parse_error}")
                return []
                
        except Exception as e:
            print(f"❌ Error getting LLM recommendations: {e}")
            return []

    def get_program_similarity_recommendations(self, user_profile, user_registrations):
        """Get program similarity recommendations - WITHOUT AI enhancement initially"""
        try:
            print(f"\n🔗 GETTING PROGRAM SIMILARITY RECOMMENDATIONS")
            
            if not user_registrations:
                print("📭 No registered programs found for similarity search")
                return []
            
            # Get list of registered program IDs
            registered_program_ids = [reg['program_id'] for reg in user_registrations]
            
            print(f"📚 Finding programs similar to {len(registered_program_ids)} registered programs:")
            for reg in user_registrations:
                print(f"   - {reg['program_title']} (ID: {reg['program_id']})")
            
            # Import the similarity function
            from pinecone_utils import find_similar_programs_by_registration
            
            # Get similar programs - NO AI enhancement here
            similar_results = find_similar_programs_by_registration(
                registered_program_ids, 
                top_k=10,
                exclude_ids=set(registered_program_ids)
            )
            
            print(f"🔍 Found {len(similar_results)} similar programs")
            
            # Just create basic recommendations WITHOUT AI enhancement
            similarity_recommendations = []
            for match in similar_results:
                if match.score > 0.7:  # High threshold for similarity
                    metadata = match.metadata
                    program_id = metadata.get("program_id")
                    similar_to_title = metadata.get('similar_to_program_title', 'your registered program')
                    
                    # Create basic recommendation WITHOUT AI enhancement
                    similarity_rec = {
                        "program_id": program_id,
                        "title": metadata.get("title"),
                        "category": metadata.get("category"),
                        "skills_required": metadata.get("skills_required"),
                        "cost": metadata.get("cost"),
                        "start_date": metadata.get("start_date"),
                        "end_date": metadata.get("end_date"),
                        "score": round(match.score, 3),
                        "recommendation_type": "program_similarity",
                        "is_registered": False,
                        "similarity_score": match.score,
                        "similarity_info": {
                            "message": f"Similar to '{similar_to_title}'",
                            "similar_to_program_title": similar_to_title,
                            "explanation": f"Since you registered for '{similar_to_title}', we think you'd be interested in this similar program."
                        }
                        # NO llm_reasoning here - will be added later for selected items only
                    }
                    
                    similarity_recommendations.append(similarity_rec)
                    print(f"   ✅ Added similarity rec: {metadata.get('title')} (Score: {match.score:.3f})")
            
            # Sort by similarity score and return top 3
            similarity_recommendations.sort(key=lambda x: x['score'], reverse=True)
            top_similarity = similarity_recommendations[:3]
            
            print(f"🎯 Generated {len(top_similarity)} basic similarity recommendations (no AI enhancement yet)")
            return top_similarity
            
        except Exception as e:
            print(f"❌ Error getting program similarity recommendations: {e}")
            return []

    def get_enhanced_collaborative_recommendations(self, user_profile, user_registrations, collaborative_recs):
        """Get collaborative recommendations - WITHOUT AI enhancement initially"""
        try:
            print(f"\n👥 GETTING COLLABORATIVE RECOMMENDATIONS")
            
            if not collaborative_recs:
                print("📭 No collaborative recommendations to process")
                return []
            
            # Just create basic collaborative recommendations WITHOUT AI enhancement
            collaborative_recommendations = []
            
            for collab_rec in collaborative_recs:
                try:
                    vector_id = f"program-{collab_rec['program_id']}"
                    fetch_response = self.index.fetch(ids=[vector_id])
                    
                    if vector_id in fetch_response.vectors:
                        metadata = fetch_response.vectors[vector_id].metadata
                        
                        # Create basic recommendation WITHOUT AI enhancement
                        collab_recommendation = {
                            "program_id": metadata.get("program_id"),
                            "title": metadata.get("title"),
                            "category": metadata.get("category"),
                            "skills_required": metadata.get("skills_required"),
                            "cost": metadata.get("cost"),
                            "start_date": metadata.get("start_date"),
                            "end_date": metadata.get("end_date"),
                            "score": collab_rec['collaborative_score'] / 10,
                            "recommendation_type": "collaborative_llm",
                            "is_registered": False,
                            "collaborative_info": {
                                "users_registered": collab_rec.get('users_registered', 0),
                                "message": "Users similar to you have registered for this program"
                            }
                            # NO llm_reasoning here - will be added later for selected items only
                        }
                        collaborative_recommendations.append(collab_recommendation)
                        print(f"   ✅ Added collaborative rec: {metadata.get('title')}")
                        
                except Exception as e:
                    print(f"Error fetching program details for {collab_rec['program_id']}: {e}")
                    continue
            
            print(f"🎯 Generated {len(collaborative_recommendations)} basic collaborative recommendations (no AI enhancement yet)")
            return collaborative_recommendations
            
        except Exception as e:
            print(f"❌ Error getting collaborative recommendations: {e}")
            return []

    def enhance_final_recommendations(self, user_profile, recommendations):
        """Enhance ONLY the final selected recommendations with AI insights"""
        try:
            print(f"\n🤖 ENHANCING FINAL {len(recommendations)} RECOMMENDATIONS WITH AI...")
            
            enhanced_recommendations = []
            
            for rec in recommendations:
                # Skip if already has LLM reasoning (for llm_powered type)
                if rec.get('llm_reasoning'):
                    enhanced_recommendations.append(rec)
                    print(f"   ✅ Skipped (already enhanced): {rec['title']}")
                    continue
                
                # Enhance with AI for consistency
                program_data = {
                    'title': rec.get('title'),
                    'category': rec.get('category'),
                    'skills_required': rec.get('skills_required', '')
                }
                
                # Determine source for enhancement
                source_map = {
                    'program_similarity': 'program similarity',
                    'collaborative_llm': 'collaborative filtering',
                    'profile_match': 'profile matching'
                }
                
                recommendation_source = source_map.get(rec['recommendation_type'], 'recommendation')
                
                # Get AI enhancement
                ai_enhancement = self.enhance_recommendation_with_ai(
                    user_profile, 
                    program_data, 
                    recommendation_source
                )
                
                # Add AI enhancement to the recommendation
                rec['llm_reasoning'] = {
                    "reason": ai_enhancement.get('recommendation_reason', ''),
                    "skills_gained": ai_enhancement.get('skills_gained', ''),
                    "career_impact": ai_enhancement.get('career_impact', ''),
                    "urgency": ai_enhancement.get('urgency', 'medium')
                }
                
                enhanced_recommendations.append(rec)
                print(f"   ✅ Enhanced: {rec['title']}")
            
            print(f"🎯 Enhanced {len(enhanced_recommendations)} final recommendations with AI insights")
            return enhanced_recommendations
            
        except Exception as e:
            print(f"❌ Error enhancing final recommendations: {e}")
            return recommendations  # Return original if enhancement fails

    def get_hybrid_recommendations(self, user_profile, user_registrations, collaborative_recs):
        """Enhanced hybrid recommendations: 3 AI + 2 Profile Match for new users"""
        try:
            print(f"\n🔄 GENERATING HYBRID RECOMMENDATIONS (EFFICIENT AI ENHANCEMENT)")
            
            all_recommendations = []
            
            # 1. Get LLM recommendations (3 items) - Already enhanced
            print("\n🤖 Getting LLM recommendations...")
            llm_recommendations = self.get_llm_recommendations(user_profile, user_registrations)
            llm_program_ids = {rec['program_id'] for rec in llm_recommendations}
            all_recommendations.extend(llm_recommendations)
            
            # 2. Get program similarity recommendations (if user has registrations) - NOT enhanced yet
            if user_registrations:
                print("\n🔗 Getting program similarity recommendations...")
                similarity_recommendations = self.get_program_similarity_recommendations(user_profile, user_registrations)
                
                # Filter out duplicates from LLM recommendations
                unique_similarity_recs = []
                for rec in similarity_recommendations:
                    if rec['program_id'] not in llm_program_ids:
                        unique_similarity_recs.append(rec)
                
                print(f"   📋 Program similarity (after deduplication): {len(unique_similarity_recs)}")
                all_recommendations.extend(unique_similarity_recs)
            else:
                print("\n🔗 No registered programs - getting profile match recommendations...")
                
                # For new users, get 2 basic profile match recommendations - NOT enhanced yet
                additional_search_results = self.get_enhanced_search_results(user_profile, top_k=25)
                existing_program_ids = {rec['program_id'] for rec in all_recommendations}
                
                profile_match_count = 0
                for search_result in additional_search_results:
                    if (search_result['program_id'] not in existing_program_ids and 
                        profile_match_count < 2 and 
                        search_result['similarity_score'] > 0.65):
                        
                        # Create basic profile match WITHOUT AI enhancement
                        profile_match_rec = {
                            "program_id": search_result['program_id'],
                            "title": search_result['title'],
                            "category": search_result['category'],
                            "skills_required": search_result.get('skills_required', ''),
                            "cost": search_result.get('cost', 0),
                            "start_date": search_result.get('start_date', ''),
                            "end_date": search_result.get('end_date', ''),
                            "score": search_result['similarity_score'],
                            "recommendation_type": "profile_match",
                            "is_registered": False,
                            "similarity_score": search_result['similarity_score'],
                            "match_info": {
                                "message": "Matches your profile and interests",
                                "explanation": f"This program aligns well with your role as {user_profile.get('role', 'professional')} and interests in {user_profile.get('interests', 'your field')}"
                            }
                            # NO llm_reasoning here - will be added later
                        }
                        
                        all_recommendations.append(profile_match_rec)
                        profile_match_count += 1
                        print(f"   ✅ Added profile match: {search_result['title']} (Score: {search_result['similarity_score']:.3f})")
                
                print(f"   📊 Added {profile_match_count} basic profile match recommendations (no AI enhancement yet)")
            
            # 3. Get collaborative recommendations (if space allows) - NOT enhanced yet
            print("\n👥 Getting collaborative recommendations...")
            existing_program_ids = {rec['program_id'] for rec in all_recommendations}
            
            # Filter collaborative recs to avoid duplicates
            unique_collaborative_recs = []
            for collab_rec in collaborative_recs:
                if collab_rec['program_id'] not in existing_program_ids and len(all_recommendations) < 5:
                    unique_collaborative_recs.append(collab_rec)
            
            collaborative_recommendations = self.get_enhanced_collaborative_recommendations(
                user_profile, user_registrations, unique_collaborative_recs
            )
            
            print(f"   👥 Collaborative (after deduplication): {len(collaborative_recommendations)}")
            all_recommendations.extend(collaborative_recommendations)
            
            # 4. NOW enhance ONLY the final selected recommendations with AI
            print(f"\n🎯 SELECTING TOP 5 AND ENHANCING WITH AI...")
            
            # Sort by recommendation type priority and score
            def get_priority(rec):
                type_priority = {
                    'program_similarity': 1,    # Highest priority - similar to registered programs
                    'llm_powered': 2,          # AI recommendations  
                    'profile_match': 3,        # Enhanced profile matches
                    'collaborative_llm': 4     # Social recommendations
                }
                return (type_priority.get(rec['recommendation_type'], 5), -rec.get('score', 0))
            
            all_recommendations.sort(key=get_priority)
            
            # Take top 5 and THEN enhance them
            top_5_recommendations = all_recommendations[:5]
            
            # Final AI enhancement for selected recommendations only
            final_enhanced_recommendations = self.enhance_final_recommendations(user_profile, top_5_recommendations)
            
            # Final summary
            print(f"\n🎯 FINAL EFFICIENT HYBRID RECOMMENDATIONS SUMMARY:")
            llm_count = len([r for r in final_enhanced_recommendations if r['recommendation_type'] == 'llm_powered'])
            similarity_count = len([r for r in final_enhanced_recommendations if r['recommendation_type'] == 'program_similarity'])
            collaborative_count = len([r for r in final_enhanced_recommendations if r['recommendation_type'] == 'collaborative_llm'])
            profile_match_count = len([r for r in final_enhanced_recommendations if r['recommendation_type'] == 'profile_match'])
            
            print(f"   🤖 LLM-powered (AI Analysis): {llm_count}")
            print(f"   🔗 Program Similarity (AI Enhanced): {similarity_count}")
            print(f"   🎯 Profile Match (AI Enhanced): {profile_match_count}")
            print(f"   👥 Collaborative (AI Enhanced): {collaborative_count}")
            print(f"   🔄 Total: {len(final_enhanced_recommendations)}")
            
            # Print final order
            print(f"\n📋 FINAL RECOMMENDATION ORDER (Efficiently Enhanced):")
            for i, rec in enumerate(final_enhanced_recommendations):
                type_label = {
                    'program_similarity': '🔗 Similar to Registered',
                    'llm_powered': '🤖 AI Recommended',
                    'profile_match': '🎯 Profile Match',
                    'collaborative_llm': '👥 Social Proof'
                }.get(rec['recommendation_type'], '📋 Other')
                
                print(f"   {i+1}. {rec['title']} ({type_label}) - Score: {rec.get('score', 0):.3f}")
            
            return final_enhanced_recommendations
            
        except Exception as e:
            print(f"❌ Error generating hybrid recommendations: {e}")
            return []

# Global instance
llm_engine = LLMRecommendationEngine()
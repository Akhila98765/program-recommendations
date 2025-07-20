import os
from nomic import embed
from pinecone import Pinecone
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV") or "gcp-starter"
INDEX_NAME = "programs"

# Fix: Use the loaded variable instead of os.environ
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV) 
index = pc.Index(INDEX_NAME)

def embed_text(text: str, input_type: str = "search_query"):
    """
    Embed text using Nomic's embedding model
    input_type can be "search_query" or "search_document"
    """
    output = embed.text(
        texts=[f"{input_type}: {text}"],
        model='nomic-embed-text-v1.5'
    )
    return output['embeddings'][0]

def search_similar_programs(query: str, filters: dict = None, top_k: int = 10):  # Changed from 5 to 10
    print(f"🔍 Search query: {query}")
    print(f"🔧 Filters: {filters}")
    print(f"📊 Top K: {top_k}")
    
    try:
        query_vec = embed_text(query, input_type="search_query")
        print(f"✅ Query vector generated, length: {len(query_vec)}")
        
        response = index.query(
            vector=query_vec, 
            top_k=top_k, 
            include_metadata=True, 
            filter=filters or {}
        )
        
        print(f"📋 Pinecone response: {len(response.matches)} matches")
        for i, match in enumerate(response.matches):
            print(f"   {i+1}. Score: {match.score}, ID: {match.id}")
        
        return response.matches
    
    except Exception as e:
        print(f"❌ Error in search_similar_programs: {e}")
        return []

def find_similar_programs_by_registration(registered_program_ids: list, top_k: int = 10, exclude_ids: set = None):
    """
    Find programs similar to the ones user has already registered for
    """
    if not registered_program_ids:
        return []
    
    print(f"🔍 Finding programs similar to registered programs: {registered_program_ids}")
    
    try:
        all_similar_matches = []
        
        # For each registered program, find similar ones
        for program_id in registered_program_ids:
            vector_id = f"program-{program_id}"
            
            # Get the vector for this program
            fetch_response = index.fetch(ids=[vector_id])
            
            if vector_id in fetch_response.vectors:
                program_vector = fetch_response.vectors[vector_id].values
                
                # Get the registered program's metadata for better context
                registered_program_metadata = fetch_response.vectors[vector_id].metadata
                registered_program_title = registered_program_metadata.get('title', f'Program {program_id}')
                
                # Find similar programs using this vector
                similar_response = index.query(
                    vector=program_vector,
                    top_k=top_k + 5,
                    include_metadata=True
                )
                
                print(f"📋 Found {len(similar_response.matches)} similar programs for {registered_program_title}")
                
                # Add similarity info and filter
                for match in similar_response.matches:
                    match_program_id = match.metadata.get("program_id")
                    
                    # Skip if it's the same program or already registered
                    if (match_program_id == program_id or 
                        (exclude_ids and match_program_id in exclude_ids)):
                        continue
                    
                    # Add detailed reference to which program this is similar to
                    match.metadata['similar_to_program_id'] = program_id
                    match.metadata['similar_to_program_title'] = registered_program_title
                    match.metadata['similarity_reason'] = f"Similar to '{registered_program_title}'"
                    match.metadata['recommendation_explanation'] = f"You registered for '{registered_program_title}', so you might be interested in this similar program"
                    
                    all_similar_matches.append(match)
        
        # Remove duplicates and sort by similarity score
        unique_matches = {}
        for match in all_similar_matches:
            program_id = match.metadata.get("program_id")
            if program_id not in unique_matches or match.score > unique_matches[program_id].score:
                unique_matches[program_id] = match
        
        # Sort by score and return top results
        sorted_matches = sorted(unique_matches.values(), key=lambda x: x.score, reverse=True)
        
        print(f"✅ Returning {len(sorted_matches[:top_k])} unique similar programs")
        return sorted_matches[:top_k]
        
    except Exception as e:
        print(f"❌ Error finding similar programs by registration: {e}")
        return []

def get_program_details(program_id: str):
    """Get full program details from Pinecone by program ID"""
    try:
        vector_id = f"program-{program_id}"
        print(f"🔍 Fetching details for program: {vector_id}")
        
        # Fetch from Pinecone
        fetch_response = index.fetch(ids=[vector_id])
        
        if vector_id in fetch_response.vectors:
            metadata = fetch_response.vectors[vector_id].metadata
            print(f"✅ Found program details: {metadata.get('title', 'Unknown')}")
            return {
                "program_id": metadata.get("program_id"),
                "title": metadata.get("title"),
                "category": metadata.get("category"),
                "skills_required": metadata.get("skills_required"),
                "cost": metadata.get("cost"),
                "start_date": metadata.get("start_date"),
                "end_date": metadata.get("end_date")
            }
        else:
            print(f"❌ Program {vector_id} not found in Pinecone")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching program details: {e}")
        return None

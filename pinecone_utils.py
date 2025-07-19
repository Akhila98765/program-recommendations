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

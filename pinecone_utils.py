import os
import cohere
from pinecone import Pinecone
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV") or "gcp-starter"
INDEX_NAME = "programs"

co = cohere.Client(COHERE_API_KEY)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter") 
index = pc.Index(INDEX_NAME)

def embed_text(text: str, input_type: str = "search_query"):
    response = co.embed(texts=[text], model="embed-english-v3.0", input_type=input_type)
    return response.embeddings[0]

def search_similar_programs(query: str, filters: dict = None, top_k: int = 5):
    query_vec = embed_text(query, input_type="search_query")
    response = index.query(vector=query_vec, top_k=top_k, include_metadata=True, filter=filters or {})
    return response.matches

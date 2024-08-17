# pcone_tester.py
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
PINECONE_SPEC_CLOUD = os.getenv('PINECONE_SPEC_CLOUD')
PINECONE_SPEC_REGION = os.getenv('PINECONE_SPEC_REGION')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Pinecone client, SentenceTransformer model, and Groq client
print("Initializing Pinecone client, model, and Groq client...")
pc = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = Groq(api_key=GROQ_API_KEY)
MODEL = 'llama3-groq-70b-8192-tool-use-preview'

# Connect to or create the Pinecone index
index_name = "robo-rag"
dimension = 384
metric = "cosine"
spec = ServerlessSpec(cloud="aws", region="us-east-1")

index = pc.Index(index_name)  # Correctly referencing the index here

# Function to query the Pinecone index
def query_pinecone(query_text, top_k=3):
    # Encode the query text into an embedding
    query_embedding = model.encode(query_text).tolist()
    
    # Query the Pinecone index to find the top_k most similar vectors
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="ns1"
    )

    return results

# Example query
query_text = "What can admins do?"
results = query_pinecone(query_text, top_k=3)

# Display results
for match in results['matches']:
    print(f"Score: {match['score']}")
    print(f"URL: {match['metadata']['url']}")
    print(f"Title: {match['metadata']['title']}")
    print(f"Content: {match['metadata']['content']}")
    print("\n---\n")

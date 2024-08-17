# pcone_tester.py

# This script tests the Pinecone database connection by querying the DB with a sample query and
# printing the metadata of the top matching results.

import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
PINECONE_SPEC_CLOUD = os.getenv('PINECONE_SPEC_CLOUD')
PINECONE_SPEC_REGION = os.getenv('PINECONE_SPEC_REGION')

# Initialize Pinecone client and SentenceTransformer model
pc = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Connect to or create the Pinecone index
index_name = PINECONE_INDEX_NAME
dimension = 384
metric = "cosine"
spec = ServerlessSpec(cloud=PINECONE_SPEC_CLOUD, region=PINECONE_SPEC_REGION)
index = pc.Index(index_name)

# Function to query the Pinecone index
def query_pinecone(query_text, top_k=3):
    query_embedding = model.encode(query_text).tolist()
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
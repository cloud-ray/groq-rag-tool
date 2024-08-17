# pcone_tester.py
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Initialize Pinecone client and model
pc = Pinecone(api_key="cb1fc5e7-0b9e-46a2-9bef-8a12a33f6428")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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

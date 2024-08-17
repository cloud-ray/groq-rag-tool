# pcone_tester.py
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq

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

index = pc.Index(index_name)

# Define top_k value
TOP_K = 3

# Function to query the Pinecone index
def query_pinecone(query_text, top_k=TOP_K):
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
query_text = """
I'm experiencing issues with mislabeled bounding boxes when exporting a dataset from LabelImg to Roboflow and then using it in a Python function. The labels appear to be misplaced both in Roboflow and in my Python code.

**Step-by-Step Breakdown**

1. **Initial Label Creation**: I created the labels for my dataset using the LabelImg Python library.
2. **Label Import to Roboflow**: I imported the labeled dataset into the Roboflow environment to correct some labels.
3. **Label Correction in Roboflow**: I made some corrections to the labels in Roboflow.
4. **Exporting Labels from Roboflow**: I exported the corrected labels from Roboflow.
5. **Importing Labels into Python Function**: I imported the exported labels into my Python function.
6. **Mislabeled Bounding Boxes**: The bounding boxes appear to be misplaced both in the Roboflow interface and in my Python code.

How do I fix this?
"""

results = query_pinecone(query_text)

# Construct the prompt
prompt = f"""
You are an assistant for question-answering tasks. Use the following Documents of retrieved knowledge base Context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
If applicable, return the most relevant URL for the answer in this format:
To learn more, visit: <url>
"""

# Add the query to the prompt
prompt += f"\n\n**Question:** {query_text}\n\n"

# Add retrieved chunks to the prompt
prompt += "**Context:**\n\n"
for match in results['matches']:
    chunk_url = match['metadata']['url']
    chunk_title = match['metadata']['title']
    chunk_text = match['metadata']['content'] 

    prompt += f"**Document:** {chunk_title}\n"
    prompt += f"{chunk_url}\n\n"
    prompt += f"{chunk_text}\n\n"

# print(f"Prompt:\n{prompt}")


# @observe(as_type="generation")
def generate_response(prompt, query_text):
    model = "llama-3.1-8b-instant"

    rag_chat_completion = llm.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query_text},
        ],
        model=model,
        temperature=0,
        max_tokens=250
    )

    # Extract usage details
    usage = rag_chat_completion.usage

    return rag_chat_completion

# Example usage
rag_chat_completion = generate_response(prompt, query_text)

# Extract and print the content
response_content = rag_chat_completion.choices[0].message.content
print(f"Response Content:\n{response_content}")

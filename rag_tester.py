# rag_tester.py

# This script tests the Groq model by querying it with a sample text, 
# retrieving relevant documents from a Pinecone knowledge base, and printing the response.

import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq
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
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
llm = Groq(
    api_key=GROQ_API_KEY,
)
llm_model = 'llama-3.1-8b-instant'

# Define top_k value
TOP_K = 3

# Function to query the Pinecone index
def query_pinecone(query_text, top_k=TOP_K):
    query_embedding = model.encode(query_text).tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="ns1"
    )
    return results

# Function to generate response
def generate_response(prompt, query_text):
    model = llm_model
    rag_chat_completion = llm.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query_text},
        ],
        model=model,
        temperature=0,
        max_tokens=250
    )
    usage = rag_chat_completion.usage
    return rag_chat_completion

def main():
    query_text = """
    How do I export my dataset?
    """
    results = query_pinecone(query_text)

    prompt = f"""
    You are an assistant for question-answering tasks. Use the following Documents of retrieved knowledge base Context to answer the question. 
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    """

    prompt += f"\n\n**Question:** {query_text}\n\n"
    prompt += "**Context:**\n\n"
    for match in results['matches']:
        chunk_url = match['metadata']['url']
        chunk_title = match['metadata']['title']
        chunk_text = match['metadata']['content'] 

        prompt += f"**Document:** {chunk_title}\n"
        prompt += f"{chunk_url}\n\n"
        prompt += f"{chunk_text}\n\n"

    rag_chat_completion = generate_response(prompt, query_text)
    response_content = rag_chat_completion.choices[0].message.content
    print(f"Response Content:\n{response_content}")

    usage = rag_chat_completion.usage
    print("\nUsage Details:")
    print(f"Completion Tokens: {usage.completion_tokens}")
    print(f"Prompt Tokens: {usage.prompt_tokens}")
    print(f"Total Tokens: {usage.total_tokens}")
    print(f"Completion Time: {usage.completion_time:.4f} seconds")
    print(f"Prompt Time: {usage.prompt_time:.4f} seconds")
    print(f"Queue Time: {usage.queue_time:.4f} seconds")
    print(f"Total Time: {usage.total_time:.4f} seconds")

if __name__ == "__main__":
    main()
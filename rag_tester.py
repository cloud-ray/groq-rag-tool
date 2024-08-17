# pcone_tester.py
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq

llm = Groq(
    api_key='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',
)

# Initialize Pinecone client and model
pc = Pinecone(api_key="cb1fc5e7-0b9e-46a2-9bef-8a12a33f6428")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Connect to or create the Pinecone index
index_name = "robo-rag"
# dimension = 384
# metric = "cosine"
spec = ServerlessSpec(cloud="aws", region="us-east-1")
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
How do I export my dataset?
"""

results = query_pinecone(query_text)

# Construct the prompt
prompt = f"""
You are an assistant for question-answering tasks. Use the following Documents of retrieved knowledge base Context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
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

    # Update Langfuse context with token details and other usage metrics
    # langfuse_context.update_current_observation(
    #     usage={
    #         "completion_tokens": usage.completion_tokens,
    #         "prompt_tokens": usage.prompt_tokens,
    #         "total_tokens": usage.total_tokens,
    #         "completion_time": usage.completion_time,
    #         "prompt_time": usage.prompt_time,
    #         "queue_time": usage.queue_time,
    #         "total_time": usage.total_time
    #     },
    #     model=model,
    #     input=query_manual,
    #     output=rag_chat_completion.choices[0].message.content
    # )

    return rag_chat_completion

# Example usage
rag_chat_completion = generate_response(prompt, query_text)

# Extract and print the content
response_content = rag_chat_completion.choices[0].message.content
print(f"Response Content:\n{response_content}")

# Extract and print usage details
usage = rag_chat_completion.usage
print("\nUsage Details:")
print(f"Completion Tokens: {usage.completion_tokens}")
print(f"Prompt Tokens: {usage.prompt_tokens}")
print(f"Total Tokens: {usage.total_tokens}")
print(f"Completion Time: {usage.completion_time:.4f} seconds")
print(f"Prompt Time: {usage.prompt_time:.4f} seconds")
print(f"Queue Time: {usage.queue_time:.4f} seconds")
print(f"Total Time: {usage.total_time:.4f} seconds")

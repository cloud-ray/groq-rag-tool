# main.py
import os
from langfuse.decorators import observe, langfuse_context

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



from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')




forum_query = """
Misslabeling bounding boxes: I’m having an issue when exporting the dataset. 
I originally created the labels on Labelimg, a Python library, then I imported them to the roboflow environment to correct some labels. 
Then my problems start. When I export the labels, they appear to be misplaced both in roboflow and in my function call in Python. 
"""

forum_system_prompt = """
Carefully review the user comment, question or problem. 
Understand what the user is really asking, and rewrite/rephrase the query in extremely explicit steps and clear language.
Expected format:
1. Problem Statement
2. Step-by-Step Breakdown
"""

@observe(as_type="generation")
def generate_chat_completion(forum_system_prompt, forum_query):
    model = "llama-3.1-8b-instant"
    
    chat_completion = llm.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": forum_system_prompt
            },
            {
                "role": "user",
                "content": forum_query,
            }
        ],
        model=model,
        temperature=0
    )

    # Extract usage details
    usage = chat_completion.usage

    # Update Langfuse context with token details and other usage metrics
    langfuse_context.update_current_observation(
        usage={
            "completion_tokens": usage.completion_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "total_tokens": usage.total_tokens,
            "completion_time": usage.completion_time,
            "prompt_time": usage.prompt_time,
            "queue_time": usage.queue_time,
            "total_time": usage.total_time
        },
        # tags=["query_rewrite"],
        model=model,
        input=forum_query,
        output=chat_completion.choices[0].message.content
    )

    # Return both content and the completion object
    return chat_completion.choices[0].message.content, chat_completion

# Example usage
query, chat_completion = generate_chat_completion(forum_system_prompt, forum_query)
print(query)

# Print the token details
usage = chat_completion.usage
print("Token Details:")
print(f"Completion Tokens: {usage.completion_tokens}")
print(f"Prompt Tokens: {usage.prompt_tokens}")
print(f"Total Tokens: {usage.total_tokens}")
print(f"Completion Time: {usage.completion_time:.2f} seconds")
print(f"Prompt Time: {usage.prompt_time:.2f} seconds")
print(f"Queue Time: {usage.queue_time:.2f} seconds")
print(f"Total Time: {usage.total_time:.2f} seconds")




query_manual = """
What can I do as an admin?
"""


# Generate the embedding for the query
query_embedding = model.encode(query_manual).tolist()

print(f"Query Embedding: {query_embedding}")




# Perform similarity search in ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# Iterate over the top results
for i in range(len(results['ids'][0])):
    print(f"Result {i + 1}:")
    print(f"Chunk ID: {results['ids'][0][i]}")
    print(f"Metadata: {results['metadatas'][0][i]}")
    print(f"Distance: {results['distances'][0][i]}")
    print("\n---\n")






# Construct the prompt
prompt = f"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
"""

# Add the query to the prompt
prompt += f"\n\n**Question:** {query_manual}\n\n"

# Add retrieved chunks to the prompt
prompt += "**Context:**\n\n"
for i in range(len(results['ids'][0])):
    chunk_id = results['ids'][0][i]  # Get the chunk ID
    chunk_metadata = results['metadatas'][0][i]  # Get the metadata
    chunk_url = chunk_metadata['url']
    chunk_index = chunk_metadata['chunk_index']

    # Find the chunk text in the test_data list
    for item in test_data:
        if item['url'] == chunk_url:
            chunk_text = item['chunks'][chunk_index]
            break

    prompt += f"**Document {i+1}: {chunk_metadata['title']}**\n"
    prompt += f"{chunk_url}\n\n"
    prompt += f"{chunk_text}\n\n"

print(f"Prompt:\n{prompt}")





# Define the generate_response function with observation
@observe(as_type="generation")
def generate_response(prompt, query_manual):
    model = "llama-3.1-8b-instant"

    rag_chat_completion = llm.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query_manual},
        ],
        model=model,
        temperature=0,
        max_tokens=250
    )

    # Extract usage details
    usage = rag_chat_completion.usage

    # Update Langfuse context with token details and other usage metrics
    langfuse_context.update_current_observation(
        usage={
            "completion_tokens": usage.completion_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "total_tokens": usage.total_tokens,
            "completion_time": usage.completion_time,
            "prompt_time": usage.prompt_time,
            "queue_time": usage.queue_time,
            "total_time": usage.total_time
        },
        model=model,
        input=query_manual,
        output=rag_chat_completion.choices[0].message.content
    )

    # Return both the generated answer and the completion object
    return rag_chat_completion.choices[0].message.content, rag_chat_completion
    print('RAG CHAT COMPLETION')
    print(rag_chat_completion)


# Example usage
answer, rag_chat_completion = generate_response(prompt, query_manual)  # Capture both the answer and the completion object
print("\nGenerated Answer:")
print(answer)

# Print additional metadata about the chunk (assuming `results` is already defined)
chunk_id = results['ids'][0][0]  # Get the first chunk ID
chunk_metadata = results['metadatas'][0][0]  # Get the first metadata
url = chunk_metadata['url']
print(f"To learn more, visit: {url}")
print()

# Print the token details
usage = rag_chat_completion.usage
print("Token Details:")
print(f"Completion Tokens: {usage.completion_tokens}")
print(f"Prompt Tokens: {usage.prompt_tokens}")
print(f"Total Tokens: {usage.total_tokens}")
print(f"Completion Time: {usage.completion_time:.2} seconds")
print(f"Prompt Time: {usage.prompt_time:.2} seconds")
print(f"Queue Time: {usage.queue_time:.2} seconds")
print(f"Total Time: {usage.total_time:.2} seconds")











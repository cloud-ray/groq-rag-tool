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

# Create or connect to a Pinecone index
index_name = "robo-rag"
dimension = 384
metric = "cosine"
spec = ServerlessSpec(cloud="aws", region="us-east-1")

# Ensure the index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=dimension, 
        metric=metric,
        spec=spec
    )

index = pc.Index(index_name)

def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    sections = content.split('--------------------------------------------------------------------------------')
    extracted_data = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        url_start = section.find('URL:') + len('URL:')
        url_end = section.find('Title:')
        url = section[url_start:url_end].strip()

        title_start = section.find('Title:') + len('Title:')
        title_end = section.find('\n', title_start)
        title = section[title_start:title_end].strip()

        content_start = title_end
        content = section[content_start:].strip()

        extracted_data.append({
            'url': url,
            'title': title,
            'content': content
        })

    return extracted_data

def chunk_text(text, chunk_size=512, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def add_to_pinecone(data):
    for item in data:
        print(f"Processing URL: {item['url']}")
        print(f"Title: {item['title']}")
        print(f"Number of Chunks: {len(item['chunks'])}")

        for i, chunk in enumerate(item['chunks']):
            chunk_embedding = model.encode(chunk).tolist()
            # chunk_id = f"{item['url']}#{i}"
            chunk_id = f"{item['url'].replace('/', '_')}#chunk{i}"
            
            # Print summary of chunk details
            print(f"Chunk ID: {chunk_id}")
            print(f"Chunk {i} Content (first 100 chars): {chunk[:100]}")
            print(f"Chunk {i} Embedding (first 10 values): {chunk_embedding[:10]}")
            print("\n---\n")
            
            # Upsert vector to Pinecone index with metadata, including content
            index.upsert(
                vectors=[
                    (chunk_id, chunk_embedding, {
                        "url": item['url'], 
                        "title": item['title'], 
                        "content": chunk 
                    })
                ],
                namespace="ns1"
            )
            print(f"Added chunk {i} for URL '{item['url']}' to Pinecone.")


def process_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                data = parse_file(file_path)
                for item in data:
                    item['chunks'] = chunk_text(item['content'])
                add_to_pinecone(data)
                print(f"Processed and added data from {file_path} to Pinecone.")
                print("\n===\n")

# Example usage:
directory_path = "extracted_contents/TO_USE"
process_directory(directory_path)

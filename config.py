import os

# Langfuse Configuration
LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY', 'sk-lf-112de8e9-b2f8-4484-a607-4058999af1b4')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-253d1fbb-858a-4121-948b-28d29dbf4f58')
LANGFUSE_HOST = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')

# Groq Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')

# ChromaDB Configuration
COLLECTION_NAME = 'knowledge_base_1'

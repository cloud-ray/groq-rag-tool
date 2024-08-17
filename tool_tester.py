import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import json
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

# Connect to or create the Pinecone index
print("Connecting to Pinecone index...")
index_name = "robo-rag"
spec = ServerlessSpec(cloud="aws", region="us-east-1")
index = pc.Index(index_name)

def query_pinecone(query_text, top_k=3):
    try:
        print(f"Querying Pinecone with text: '{query_text}' and top_k: {top_k}...")
        
        # Encode the query text into an embedding
        query_embedding = model.encode(query_text).tolist()
        # print(f"Query embedding: {query_embedding[:5]}... (truncated for display)")

        # Query the Pinecone index to find the top_k most similar vectors
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="ns1"
        )

        # Convert the results to a JSON-serializable format
        results_dict = {
            "matches": [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match["metadata"]
                }
                for match in results["matches"]
            ],
            "namespace": results["namespace"],
            "usage": results.get("usage")
        }

        # Remove the 'usage' field from the response if it exists
        if 'usage' in results_dict:
            del results_dict['usage']

        # print("Query results:", results_dict)

        return json.dumps(results_dict)  # Serialize the results dictionary to JSON

    except Exception as e:
        print(f"Error during Pinecone query: {str(e)}")
        return json.dumps({"error": str(e)})


# Define the conversation with the tool
def run_conversation(user_prompt):
    print(f"Running conversation with user prompt: '{user_prompt}'...")
    messages = [
        {
            "role": "system",
            "content": "You are a Roboflow knowledge base assistant that retrieves information from a Pinecone vector database and provides a clear and detailed answer to the user's question."
        },
        {
            "role": "user",
            "content": user_prompt + " (Please search the database using the available tools.)",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_pinecone",
                "description": "Query the Pinecone vector database to retrieve relevant information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": "The text query to search the Pinecone database.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "The number of top matches to retrieve.",
                            "default": 3
                        }
                    },
                    "required": ["query_text"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )
    print("Initial response received from Groq model.")
    response_message = response.choices[0].message
    print("Response message:", response_message)

    # Extract and print usage details
    usage = response.usage
    print("\nFIRST Usage Details:")
    print(f"Completion Tokens: {usage.completion_tokens}")
    print(f"Prompt Tokens: {usage.prompt_tokens}")
    print(f"Total Tokens: {usage.total_tokens}")
    print(f"Completion Time: {usage.completion_time:.4f} seconds")
    print(f"Prompt Time: {usage.prompt_time:.4f} seconds")
    # print(f"Queue Time: {usage.queue_time:.4f} seconds")
    print(f"Total Time: {usage.total_time:.4f} seconds")
    print()

    tool_calls = response_message.tool_calls
    # print("Tool calls:", tool_calls)

    if tool_calls:
        available_functions = {
            "query_pinecone": query_pinecone,
        }
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            # print(f"Calling function: {function_name}")
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            # print("Function arguments:", function_args)
            function_response = function_to_call(
                query_text=function_args.get("query_text"),
                top_k=function_args.get("top_k", 3)
            )
            # print("Function response:", function_response)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )

        # print("Second response received.")
        print("Second response message:", second_response.choices[0].message)

        # Extract and print usage details
        second_usage = second_response.usage
        print("SECOND Usage Details:")
        print(f"Completion Tokens: {second_usage.completion_tokens}")
        print(f"Prompt Tokens: {second_usage.prompt_tokens}")
        print(f"Total Tokens: {second_usage.total_tokens}")
        print(f"Completion Time: {second_usage.completion_time:.4f} seconds")
        print(f"Prompt Time: {second_usage.prompt_time:.4f} seconds")
        print(f"Total Time: {second_usage.total_time:.4f} seconds")

        return second_response.choices[0].message.content

    else:
        print("No tool calls detected.")
        return "No tool calls were made by the Groq model."

# Example user prompt
user_prompt = "How do I train a model?"

print("Final output:\n", run_conversation(user_prompt))

# user_prompt = """
# Misslabeling bounding boxes: I’m having an issue when exporting the dataset. 
# I originally created the labels on Labelimg, a Python library, then I imported them to the roboflow environment to correct some labels. 
# Then my problems start. When I export the labels, they appear to be misplaced both in roboflow and in my function call in Python. 
# """

# fuse_tester.py

import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq
from langfuse.decorators import observe, langfuse_context
import uuid

# Load environment variables from a .env file
load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
PINECONE_SPEC_CLOUD = os.getenv('PINECONE_SPEC_CLOUD')
PINECONE_SPEC_REGION = os.getenv('PINECONE_SPEC_REGION')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY')
LANGFUSE_HOST = os.getenv('LANGFUSE_HOST')

# Initialize Pinecone client, SentenceTransformer model, and Groq client
print("Initializing Pinecone client, model, and Groq client...")
pc = Pinecone(api_key=PINECONE_API_KEY)
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = Groq(api_key=GROQ_API_KEY)
tool_model = 'llama3-groq-70b-8192-tool-use-preview'

# Connect to or create the Pinecone index
print("Connecting to Pinecone index...")
index_name = PINECONE_INDEX_NAME
spec = ServerlessSpec(cloud=PINECONE_SPEC_CLOUD, region=PINECONE_SPEC_REGION)
index = pc.Index(index_name)


# Function to query Pinecone
def query_pinecone(query_text, top_k=3):
    """
    Query the Pinecone vector database to retrieve relevant information.

    Parameters:
    - query_text (str): The text query to search the Pinecone database.
    - top_k (int): The number of top matches to retrieve (default is 3).

    Returns:
    - dict: A dictionary containing matches and metadata.
    """
    try:
        print(f"Querying Pinecone with text: '{query_text}' and top_k: {top_k}...")
        
        # Encode the query text into an embedding
        query_embedding = embed_model.encode(query_text).tolist()

        # Query the Pinecone index to find the top_k most similar vectors
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="ns1"
        )

        # Convert the results to a dictionary
        results_dict = {
            "matches": [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match["metadata"]
                }
                for match in results["matches"]
            ],
            "namespace": results["namespace"]
        }

        return results_dict

    except Exception as e:
        print(f"Error during Pinecone query: {str(e)}")
        return {"error": str(e)}

# Function to generate the first response from Groq
@observe(as_type="generation")
def generate_first_response(user_prompt):
    """
    Generate the first response from the Groq model using the user prompt.

    Parameters:
    - user_prompt (str): The prompt to send to the Groq model.

    Returns:
    - dict: A dictionary containing the response and usage details.
    """
    print(f"Running conversation with user prompt: '{user_prompt}'...")
    messages = [
        {
            "role": "system",
            "content": "You are a knowledge base assistant that retrieves information from a Pinecone vector database."
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
        model=tool_model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )

    response_message = response.choices[0].message
    usage = response.usage

    # Update Langfuse context
    langfuse_context.update_current_observation(
        usage={
            "completion_tokens": usage.completion_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "total_tokens": usage.total_tokens,
            "completion_time": usage.completion_time,
            "prompt_time": usage.prompt_time,
            "total_time": usage.total_time
        },
        model=tool_model,
        input=messages[1]["content"],
    )

    return response_message, usage

# Function to process tool calls and aggregate results
def process_tool_calls(tool_calls):
    """
    Process the tool calls and aggregate results from Pinecone queries.

    Parameters:
    - tool_calls (list): List of tool calls from the first response.

    Returns:
    - tuple: A tuple containing aggregated results and the number of duplicates removed.
    """
    available_functions = {
        "query_pinecone": query_pinecone,
    }

    aggregated_results = []
    seen_results = set()
    duplicate_count = 0

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(
            query_text=function_args.get("query_text"),
            top_k=function_args.get("top_k", 3)
        )
        
        matches = function_response.get("matches", [])

        for match in matches:
            match_id = match["id"]
            if match_id in seen_results:
                print(f"Duplicate detected and removed: {match_id}")
                duplicate_count += 1
            else:
                seen_results.add(match_id)
                aggregated_results.append(match)

    return aggregated_results, duplicate_count

# Function to generate the second response from Groq
@observe(as_type="generation")
def generate_second_response(aggregated_results, tool_call_id):
    """
    Generate the second response from the Groq model using aggregated results.

    Parameters:
    - aggregated_results (list): List of aggregated results to include in the response.
    - tool_call_id (str): The ID of the last tool call.

    Returns:
    - dict: A dictionary containing the final response and usage details.
    """
    formatted_results = [
        f"{result.get('metadata', {}).get('title', 'No Title')}, {result.get('metadata', {}).get('url', 'No URL')}\n"
        f"{' '.join(result.get('metadata', {}).get('content', 'No Content').split())}"
        for result in aggregated_results
    ]

    final_content = "\n".join(formatted_results)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful Roboflow knowledge-base assistant. Use the following retrieved content from the knowledge base to answer the question. "
                "Break down and solve the problem step by step. Be as detailed and explicit as possible, but do not make anything up. "
                "Return the most relevant URL(s) for the answer: To learn more, visit: <url> "
            )
        },
        {
            "role": "tool",
            "name": "aggregated_results",
            "tool_call_id": tool_call_id,
            "content": final_content,
        }
    ]

    second_response = client.chat.completions.create(
        model=tool_model,
        messages=messages
    )

    second_response_message = second_response.choices[0].message
    second_usage = second_response.usage

    # Update Langfuse context
    langfuse_context.update_current_observation(
        usage={
            "completion_tokens": second_usage.completion_tokens,
            "prompt_tokens": second_usage.prompt_tokens,
            "total_tokens": second_usage.total_tokens,
            "completion_time": second_usage.completion_time,
            "prompt_time": second_usage.prompt_time,
            "total_time": second_usage.total_time
        },
        model=tool_model,
        input=messages[0]['content'] + "\n" + final_content,
        output=second_response_message.content
    )

    return second_response_message.content, second_usage

# Main function to run the conversation
@observe()
def run_conversation(user_prompt):
    """
    Run a conversation with the Groq model and handle the entire process.

    Parameters:
    - user_prompt (str): The prompt to send to the Groq model.

    Returns:
    - str: The final output from the Groq model.
    """
    try:
        # Generate the first response
        first_response_message, first_usage = generate_first_response(user_prompt)

        tool_calls = first_response_message.tool_calls

        if tool_calls:
            # Process tool calls to aggregate results
            aggregated_results, duplicate_count = process_tool_calls(tool_calls)
            print(f"Total unique results to be sent to second_response: {len(aggregated_results)}")
            print(f"Total duplicates removed: {duplicate_count}")

            # Generate the second response
            final_output, second_usage = generate_second_response(aggregated_results, tool_calls[-1].id)

            # Print usage details for debugging
            print("\nSECOND Usage Details:")
            print(f"Completion Tokens: {second_usage.completion_tokens}")
            print(f"Prompt Tokens: {second_usage.prompt_tokens}")
            print(f"Total Tokens: {second_usage.total_tokens}")
            print(f"Completion Time: {second_usage.completion_time:.4f} seconds")
            print(f"Prompt Time: {second_usage.prompt_time:.4f} seconds")
            print(f"Total Time: {second_usage.total_time:.4f} seconds")

            # Generate an 8-digit hex for the session ID
            session_id = uuid.uuid4().hex[:8]

            # Generate a 4-digit hex for the user ID
            user_id = uuid.uuid4().hex[:4]

            # Update Langfuse context
            langfuse_context.update_current_trace(
                tags=["rag", "roboflow"],
                session_id=session_id,
                user_id=user_id,
                # metadata={"key": "value"}
            )

            return final_output

        else:
            print("No tool calls detected.")
            return "No tool calls were made by the Groq model."

    finally:
        # Ensure all events are flushed and pending requests are awaited
        langfuse_context.flush()

# Example user prompt
user_prompt = "How do I add tags to my images?"

print("\nFinal output:\n" + run_conversation(user_prompt))

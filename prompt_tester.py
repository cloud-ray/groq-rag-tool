import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq

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
index_name = PINECONE_INDEX_NAME
spec = ServerlessSpec(cloud=PINECONE_SPEC_CLOUD, region=PINECONE_SPEC_REGION)
index = pc.Index(index_name)

def query_pinecone(query_text, top_k=3):
    """
    Query the Pinecone vector database to retrieve relevant information.

    Parameters:
    - query_text (str): The text query to search the Pinecone database.
    - top_k (int): The number of top matches to retrieve (default is 3).

    Returns:
    - str: JSON-serialized results containing matches and metadata.
    """
    try:
        print(f"Querying Pinecone with text: '{query_text}' and top_k: {top_k}...")
        
        # Encode the query text into an embedding
        query_embedding = model.encode(query_text).tolist()

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
            "namespace": results["namespace"]
        }

        return json.dumps(results_dict)  # Serialize the results dictionary to JSON

    except Exception as e:
        print(f"Error during Pinecone query: {str(e)}")
        return json.dumps({"error": str(e)})

def run_conversation(user_prompt):
    """
    Run a conversation with the Groq model using the given user prompt.

    Parameters:
    - user_prompt (str): The prompt to send to the Groq model.

    Returns:
    - str: The final response content from the Groq model.
    """
    print(f"Running conversation with user prompt: '{user_prompt}'...")
    messages = [
        {
            "role": "system",
            "content": "You are a Roboflow knowledge base assistant that retrieves information from a Pinecone vector database."
        },
        {
            "role": "user",
            "content": user_prompt + " (Please search the database using the available tools.)",
        }
    ]

    # Print initial messages for debugging
    # print("System message for first response:")
    # print(messages[0]["content"])
    # print("User message for first response:")
    # print(messages[1]["content"])

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

    # Generate the first response using Groq
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )

    # Extract response message and usage details
    response_message = response.choices[0].message
    usage = response.usage

    # Print usage details for debugging
    print("\nFIRST Usage Details:")
    print(f"Completion Tokens: {usage.completion_tokens}")
    print(f"Prompt Tokens: {usage.prompt_tokens}")
    print(f"Total Tokens: {usage.total_tokens}")
    print(f"Completion Time: {usage.completion_time:.4f} seconds")
    print(f"Prompt Time: {usage.prompt_time:.4f} seconds")
    print(f"Total Time: {usage.total_time:.4f} seconds")
    print()

    tool_calls = response_message.tool_calls
    aggregated_results = []
    if tool_calls:
        available_functions = {
            "query_pinecone": query_pinecone,
        }
        messages.append(response_message)
        seen_results = set()  # To track seen results and avoid duplicates

        duplicate_count = 0  # Counter for duplicate results
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                query_text=function_args.get("query_text"),
                top_k=function_args.get("top_k", 3)
            )
            
            # Deserialize the function response
            function_response_dict = json.loads(function_response)
            matches = function_response_dict.get("matches", [])

            # Print received matches for debugging
            print(f"Tool call ID {tool_call.id} returned {len(matches)} matches.")
            
            # Filter out duplicates based on URL or other unique fields
            for match in matches:
                match_id = match["id"]
                if match_id in seen_results:
                    print(f"Duplicate detected and removed: {match_id}")
                    duplicate_count += 1  # Increment duplicate count
                else:
                    seen_results.add(match_id)
                    aggregated_results.append(match)

        # Print the number of unique results being sent
        print()
        print(f"Total unique results to be sent to second_response: {len(aggregated_results)}")
        print(f"Total duplicates removed: {duplicate_count}")

        messages.append(
            {
                "role": "tool",
                "name": "aggregated_results",
                "tool_call_id": tool_call.id,  # Add tool_call_id here
                "content": json.dumps({
                    "matches": aggregated_results,  # Use aggregated results
                }),
            }
        )

        # Print the updated messages list
        # print("Messages being sent to second_response:")
        # print(messages)

        # Customize the system message for the second response
        updated_system_prompt = (
            "You are an assistant for question-answering tasks. Use the following Documents of retrieved knowledge base Context to answer the question. "
            "Be as detailed and explicit as possible, but within the realm of the knowledge you have access to. "
            "If applicable, return the most relevant URL(s) for the answer in this format: "
            "To learn more, visit: <url>"
        )
        messages[0] = {
            "role": "system",
            "content": updated_system_prompt
        }

        # Print the updated messages list
        # print("Messages after updating system message:")
        # print(messages)

        # Create the second response
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )

        # Print usage details for debugging
        second_usage = second_response.usage
        print("\nSECOND Usage Details:")
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
user_prompt = "How do I add steps to a workflow?"

print("\nFinal output:\n" + run_conversation(user_prompt))

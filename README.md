# Custom Retrieval-Augmented Generation (RAG) Pipeline
![Custom RAG Pipeline](images/custom-rag.jpg)

## Overview

This project implements a custom Retrieval-Augmented Generation (RAG) pipeline tailored for managing and querying the [Roboflow](https://roboflow.com) knowledge base. 

Using [Pinecone](https://pinecone.io), SentenceTransformers, and [Groq](https://console.groq.com/docs/tool-use), with detailed tracing via [Langfuse](https://langfuse.com/), the pipeline efficiently handles user prompts, queries a vector database for relevant information, and generates responses.

## Features

- **Custom RAG Pipeline:** Integrates Pinecone for vector-based search, SentenceTransformers for text embedding, and Groq for response generation.
- **Duplicate Handling:** Identifies and removes duplicate results from Pinecone queries to reduce token usage and processing costs.
- **Detailed Tracing:** Utilizes Langfuse to monitor and debug token usage and performance metrics.
- **Flexible Function Calls:** Custom functions are used to query the Pinecone database and manage tool calls.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/cloud-ray/groq-rag-tool.git
cd groq-rag-tool
```

### 2. Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory of the project by running the following command:
```bash
cp .env.example .env
```
Then, update the `.env` file with your actual API keys and configuration details:


**Environment Variables**

| Variable | Description | Example Value |
| --- | --- | --- |
| `PINECONE_API_KEY` | Pinecone API key | `your_pinecone_api_key` |
| `PINECONE_INDEX_NAME` | Pinecone index name | `your_index_name` |
| `PINECONE_SPEC_CLOUD` | Pinecone cloud spec | `your_cloud_spec` |
| `PINECONE_SPEC_REGION` | Pinecone region spec | `your_region_spec` |
| `GROQ_API_KEY` | Groq API key | `your_groq_api_key` |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | `your_langfuse_secret_key` |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | `your_langfuse_public_key` |
| `LANGFUSE_HOST` | Langfuse host | `your_langfuse_host` |

### 4. Test the Setup

**Note:** Because this is a custom RAG setup, there is no one-size-fits-all solution. You'll need to adapt the setup to your specific use case.

To test the setup, you'll need to follow these general steps:

1. **Gather your knowledge base content**: Collect the documents or knowledge base content that you want to use with the RAG setup.
2. **Create embeddings**: Create embeddings of your knowledge base content using a library like Pinecone. You can view the `utils/pinecone_setup.py` file for an example of how to chunk and embed files.
3. **Update the `trace_tester.py` script**: Update the `trace_tester.py` script to use your own knowledge base content and embeddings.
4. **Run the `trace_tester.py` script**: Run the `trace_tester.py` script to test the setup.

```bash
python trace_tester.py
```

**Important:** Make sure to update the `trace_tester.py` script to use your own knowledge base content and embeddings. The script is designed to be flexible, so you'll need to adapt it to your specific use case.

By following these steps, you should be able to test the RAG setup with your own knowledge base content. If you have any questions or need further guidance, feel free to ask

## Code Explanation

### `trace_tester.py`

- **Environment Setup:** Loads environment variables for API keys and configuration.
- **Initialization:** Sets up Pinecone, SentenceTransformer, and Groq clients.
- **Function Definitions:**
  - **`query_pinecone(query_text, top_k=3)`**: Queries the Pinecone vector database and returns results in JSON format.
  - **`run_conversation(user_prompt)`**: Handles user prompts, generates responses using Groq, and manages tool calls and result aggregation.
- **Duplicate Handling:**
  - **Collection of Results:** Queries Pinecone multiple times, which may return overlapping data.
  - **Tracking and Filtering:** Uses metadata to track and filter out duplicate chunks. Only unique results are kept for further processing.
  - **Formatting:** Clean and aggregate unique results to present a clear and concise response.
- **Langfuse Tracing:** Updates context with usage metrics and performance details.
- **Example Execution:** Demonstrates usage with a sample user prompt.

## Duplicate Handling

The pipeline features an advanced duplicate handling mechanism designed to manage redundant data efficiently:

- **Why It Matters:** Multiple queries to Pinecone can return overlapping chunks, leading to increased token usage and higher costs. By filtering out duplicates, the system reduces these costs and improves the response quality.
- **How It Works:**
  - **Tracking Duplicates:** Unique identifiers from Pinecone results are used to track and discard duplicates.
  - **Efficient Processing:** Only unique chunks are processed and formatted, reducing the overall token count and ensuring clearer responses.
  - **Cost and Resource Optimization:** This approach not only saves on API costs but also enhances the efficiency of the response generation process.

## Customization

- **Adding New Functions:** To integrate new functions, define them similarly to `query_pinecone` and update the `available_functions` dictionary.
- **Adjusting Response Handling:** Modify the aggregation and formatting logic in `run_conversation` to suit your needs.

## Troubleshooting

- **Environment Variables:** Ensure all required environment variables are set correctly.
- **Dependencies:** Verify that all dependencies are installed and compatible with your Python version.
- **API Errors:** Check API keys and network connectivity if you encounter issues with Pinecone or Groq.

## Contributing

Feel free to submit issues and pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgements

This project utilizes the following resources:

* [Pinecone](https://docs.pinecone.io/guides/data/manage-rag-documents) for document management and retrieval.
* [Groq](https://console.groq.com/docs/tool-use) for tool usage and integration.
* [Roboflow](https://docs.roboflow.com/) for knowledge base content and data.
* [Langfuse](https://langfuse.com/docs) for tracing and monitoring.

## Contact
For any questions or comments, please contact [Ray](https://www.linkedin.com/in/raymond-fuorry).
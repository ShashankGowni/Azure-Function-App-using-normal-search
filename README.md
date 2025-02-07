# Azure Function App using Normal Search

This project implements two Azure Function APIs:

1. **IndexDocuments**: Accepts a document link (PDF or DOCX) and indexes it into Azure AI search.
2. **QueryKnowledgeBase**: Queries the indexed documents and retrieves relevant information using OpenAI's GPT model.

## Features

### **IndexDocuments**:
- Reads documents from Azure Blob Storage.
- Splits document content into chunks.
- Indexes chunks into Azure AI search.
- Handles errors and returns a status message.

### **QueryKnowledgeBase**:
- Takes a user query and searches through the indexed content.
- Uses GPT to return relevant information or null if no relevant information is found.

## Technologies Used
- **Azure Functions**
- **Azure AI Search**
- **Azure OpenAI**
- **Python**
- **Postman**

## Environment Variables
Ensure the following environment variables are set:
- `BLOB_CONNECTION_STRING`: Azure Blob Storage connection string.
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key.
- `AZURE_SEARCH_ENDPOINT`: Azure Search API endpoint.
- `AZURE_SEARCH_API_KEY`: Azure Search API key.
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI API endpoint.
- `AZURE_SEARCH_INDEX_NAME`: Name of the Azure Search index.
- `SYSTEM_MESSAGE`: The system message for GPT to base its responses on.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/ShashankGowni/Azure-Function-App-using-normal-search
    ```

2. Navigate to the project folder:
    ```bash
    cd Azure-Function-App-using-normal-search
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set the necessary environment variables:
    - For local development, create a `.env` file or set them in the terminal.

## Running Locally
To run the function app locally, use the Azure Functions Core Tools:
```bash
func start


API Usage


1. IndexDocuments API
Request Type: POST
Endpoint: http://localhost:<port>/api/IndexDocuments
Request Body:
json
Copy
Edit
{
  "doc_link": "https://yourblobstorageurl.com/yourfile.pdf"
}
Success Response:
json
Copy
Edit
{
  "status": "COMPLETED",
  "error": null
}
Failure Response:
json
Copy
Edit
{
  "status": "FAILED",
  "error": "Document format is not supported."
}
2. QueryKnowledgeBase API
Request Type: GET
Endpoint: http://localhost:<port>/api/QueryKnowledgeBase?query=What%20is%20Azure&index_name=your_index_name
Success Response:
json
Copy
Edit
{
  "response": "Azure is a cloud computing service from Microsoft.",
  "error": null
}
Failure Response:
json
Copy
Edit
{
  "response": null,
  "error": null
}
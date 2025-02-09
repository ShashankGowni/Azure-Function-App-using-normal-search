Azure Function App using Normal Search

This project implements two Azure Function APIs:

IndexDocuments: Accepts a document link (PDF or DOCX) and indexes it into Azure AI search.

QueryKnowledgeBase: Queries the indexed documents and retrieves relevant information using OpenAI's GPT model.

Features

IndexDocuments:

Reads documents from Azure Blob Storage.

Splits document content into chunks.

Indexes chunks into Azure AI search.

Handles errors and returns a status message.

QueryKnowledgeBase:

Takes a user query and searches through the indexed content.

Uses GPT to return relevant information or null if no relevant information is found.

Technologies Used

Azure Functions

Azure AI Search

Azure OpenAI

Python

Postman

Environment Variables

Ensure the following environment variables are set:

BLOB_CONNECTION_STRING: Azure Blob Storage connection string.

AZURE_OPENAI_API_KEY: Azure OpenAI API key.

AZURE_SEARCH_ENDPOINT: Azure Search API endpoint.

AZURE_SEARCH_API_KEY: Azure Search API key.

AZURE_OPENAI_ENDPOINT: Azure OpenAI API endpoint.

AZURE_SEARCH_INDEX_NAME: Name of the Azure Search index.

SYSTEM_MESSAGE: The system message for GPT to base its responses on.

Installation

Clone this repository:

git clone https://github.com/ShashankGowni/Azure-Function-App-using-normal-search

Navigate to the project folder:

cd Azure-Function-App-using-normal-search

Install the required dependencies:

pip install -r requirements.txt

Set the necessary environment variables:

For local development, create a .env file or set them in the terminal.

Running Locally

To run the function app locally, use the Azure Functions Core Tools:

func start

API Usage

1. IndexDocuments API

Request Type: POST

Endpoint: http://localhost:<port>/api/IndexDocuments

Request Body:

{
  "doc_link": "https://yourblobstorageurl.com/yourfile.pdf"
}

Success Response:

{
  "status": "COMPLETED",
  "error": null
}

Failure Response:

{
  "status": "FAILED",
  "error": "Document format is not supported."
}

2. QueryKnowledgeBase API

Request Type: GET

Endpoint: http://localhost:<port>/api/QueryKnowledgeBase?query=What%20is%20Azure&index_name=your_index_name

Success Response:

{
  "response": "Azure is a cloud computing service from Microsoft.",
  "error": null
}

Failure Response:

{
  "response": null,
  "error": null
}


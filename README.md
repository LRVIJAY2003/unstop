# Enterprise RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that integrates with Confluence, JIRA, and Remedy, using Google's Gemini API for intelligent responses.

## Overview

This application provides a natural language interface to enterprise knowledge systems, allowing users to ask questions and get accurate answers based on internal documentation, issues, and incidents. The system uses advanced RAG techniques to combine the strengths of vector search, lexical search, and large language models.

## Features

- **Multi-source Integration**: Connect to Confluence documentation, JIRA issues, and Remedy incidents
- **Advanced RAG Implementation**: 
  - TF-IDF + SVD for efficient embeddings
  - Hybrid retrieval (vector + BM25) for better results
  - Semantic chunking for natural document segmentation
- **Interactive UI**: Modern chat interface with source references and feedback mechanisms
- **Optimized Performance**: 
  - Multi-level caching
  - Efficient document processing
  - Streaming responses for faster feedback
- **Error Handling**: Comprehensive error management and graceful degradation

## Architecture

The system follows a modular architecture:

```
enterprise_rag/
│
├── Data Sources Layer: Connects to external systems
│   ├── Confluence Integration
│   ├── JIRA Integration
│   └── Remedy Integration
│
├── RAG Engine: Core retrieval and generation
│   ├── Document Chunking
│   ├── Embedding (TF-IDF + SVD)
│   ├── Hybrid Retrieval
│   └── Gemini Integration
│
├── API Layer: REST endpoints
│
└── Web Interface: User interaction
```

## Installation

### Prerequisites

- Python 3.8+
- Access to Google Cloud with Vertex AI enabled
- Access credentials for Confluence, JIRA, and Remedy

### Setup

1. Clone the repository:
   ```bash
   git clone https://your-repository-url/enterprise_rag.git
   cd enterprise_rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables by copying the example file:
   ```bash
   cp .env.example .env
   ```

5. Edit the `.env` file with your configuration details:
   - API keys and credentials
   - Service URLs
   - System settings

## Usage

### Starting the Server

Run the application using:

```bash
python run.py
```

The server will start on the configured host and port (default: `http://localhost:5000`).

### Using the Chat Interface

1. Navigate to `http://localhost:5000` in your web browser
2. Select which data sources to query (Confluence, JIRA, Remedy, or all)
3. Enter your question in the chat input
4. View the response along with source references
5. Follow up with additional questions as needed

### API Endpoints

The system provides several REST endpoints:

- `POST /api/query` - Send a query and get a response
- `GET /api/sources` - List available data sources
- `GET /api/status` - Check connection status to data sources

Example query:

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the current open incidents?", "sources":["remedy"]}'
```

## Components

### Data Source Connectors

- **Confluence Connector**: Access spaces, pages, and attachments
- **JIRA Connector**: Query issues, comments, and project information
- **Remedy Connector**: Retrieve incidents, history, and status information

### RAG Engine

- **Chunking Engine**: Breaks documents into semantic chunks for better retrieval
- **Embedding Engine**: Creates vector representations using TF-IDF + SVD
- **Retrieval Engine**: Hybrid search combining vector similarity and BM25
- **Gemini Integration**: Interfaces with Google's Gemini LLM for response generation

### Web Interface

A modern, responsive chat interface that provides:
- Real-time streaming responses
- Source attribution and references
- Data source selection
- Conversation history

## Troubleshooting

### Common Issues

1. **Connection Problems**:
   - Ensure your credentials in `.env` are correct
   - Check network connectivity to the services
   - Verify SSL settings if using self-signed certificates

2. **Missing Results**:
   - Check that documents are indexed properly
   - Ensure query is related to content in the data sources
   - Try using different search terms

3. **Slow Performance**:
   - Consider increasing cache duration in config
   - Check network latency to external services
   - Reduce chunk size or number of results if needed

### Logs

Logs are stored in the `logs` directory. Check these for detailed error information.

## Development

### Project Structure

```
enterprise_rag/
│
├── .env                       # Environment variables
├── requirements.txt           # Project dependencies
├── config.py                  # Configuration management
├── app.py                     # Main Flask application
├── run.py                     # Application launcher
│
├── static/                    # Static files
│   ├── css/                   # Stylesheets
│   ├── js/                    # JavaScript files
│   └── images/                # UI images and icons
│
├── templates/                 # Flask templates
│
├── data_sources/              # Data source connectors
│   ├── confluence/            # Confluence integration
│   ├── jira/                  # JIRA integration
│   └── remedy/                # Remedy integration
│
├── rag_engine/                # RAG implementation
│   ├── chunking.py            # Document chunking
│   ├── embedding.py           # Embedding utilities
│   ├── retrieval.py           # Vector search and retrieval
│   ├── gemini_integration.py  # Gemini API integration
│   └── processor.py           # Main RAG processor
│
├── utils/                     # Utility functions
│
└── api/                       # API endpoints
```

### Adding New Data Sources

To add a new data source:

1. Create a new directory under `data_sources/`
2. Implement a client and connector following the existing patterns
3. Update the configuration to include the new source
4. Add source-specific routes if needed

## License

This project is proprietary and confidential.

## Acknowledgments

- Google Gemini API for language model capabilities
- FAISS for efficient vector search
- scikit-learn for TF-IDF and SVD implementation
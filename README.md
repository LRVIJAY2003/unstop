"""
Main Flask application.
"""
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from config import get_config
from src.confluence.client import ConfluenceClient
from src.remedy.client import RemedyClient
from src.rag.retriever import DocumentRetriever
from src.gemini.client import GeminiClient
from src.utils.logger import setup_logger

def create_app(config_name=None):
    """Create Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    config = get_config()
    app.config.from_object(config)
    config.init_app(app)
    
    # Setup CORS
    CORS(app)
    
    # Setup logging
    logger = setup_logger(app.config['LOG_LEVEL'])
    
    # Initialize clients
    confluence_client = ConfluenceClient(
        url=app.config['CONFLUENCE_URL'],
        username=app.config['CONFLUENCE_USER_ID'],
        api_token=app.config['CONFLUENCE_API_TOKEN'],
        space_id=app.config['CONFLUENCE_SPACE_ID']
    )
    
    remedy_client = RemedyClient(
        server=app.config['REMEDY_SERVER'],
        api_base=app.config['REMEDY_API_BASE'],
        username=app.config['REMEDY_USERNAME'],
        password=app.config['REMEDY_PASSWORD']
    )
    
    # Initialize retriever
    retriever = DocumentRetriever(
        embedding_model=app.config['EMBEDDING_MODEL'],
        chunk_size=app.config['CHUNK_SIZE'],
        chunk_overlap=app.config['CHUNK_OVERLAP'],
        cache_dir=app.config['CACHE_DIR']
    )
    
    # Initialize Gemini client
    gemini_client = GeminiClient(
        project_id=app.config['PROJECT_ID'],
        model_name=app.config['MODEL_NAME'],
        region=app.config['REGION']
    )
    
    @app.route('/')
    def index():
        """Render the main page."""
        return render_template('index.html')
    
    @app.route('/api/query', methods=['POST'])
    def query():
        """Process a query."""
        data = request.json
        query_text = data.get('query', '')
        sources = data.get('sources', ['confluence', 'remedy'])  # Default to both
        
        results = {}
        
        # Get Confluence results if requested
        if 'confluence' in sources:
            try:
                confluence_pages = confluence_client.search_content(query_text)
                confluence_chunks = retriever.chunk_and_retrieve(
                    query_text, 
                    [page['content'] for page in confluence_pages]
                )
                results['confluence'] = {
                    'raw_results': confluence_pages,
                    'relevant_chunks': confluence_chunks
                }
            except Exception as e:
                logger.error(f"Confluence error: {str(e)}")
                results['confluence'] = {'error': str(e)}
        
        # Get Remedy results if requested
        if 'remedy' in sources:
            try:
                remedy_tickets = remedy_client.search_tickets(query_text)
                remedy_chunks = retriever.chunk_and_retrieve(
                    query_text, 
                    [ticket['description'] for ticket in remedy_tickets]
                )
                results['remedy'] = {
                    'raw_results': remedy_tickets,
                    'relevant_chunks': remedy_chunks
                }
            except Exception as e:
                logger.error(f"Remedy error: {str(e)}")
                results['remedy'] = {'error': str(e)}
        
        # Prepare context for Gemini
        combined_context = []
        
        if 'confluence' in results and 'relevant_chunks' in results['confluence']:
            combined_context.extend([
                f"[Confluence] {chunk}" 
                for chunk in results['confluence']['relevant_chunks']
            ])
            
        if 'remedy' in results and 'relevant_chunks' in results['remedy']:
            combined_context.extend([
                f"[Remedy] {chunk}" 
                for chunk in results['remedy']['relevant_chunks']
            ])
        
        # Generate response using Gemini
        if combined_context:
            try:
                gemini_prompt = f"""
                The user has asked: "{query_text}"
                
                Here is relevant information from our knowledge base:
                
                {' '.join(combined_context)}
                
                Based ONLY on the information provided above, give a detailed, helpful, and accurate response.
                If the information doesn't fully address the query, acknowledge the limitations.
                Format your response clearly using markdown when appropriate.
                """
                
                response = gemini_client.generate_response_from_prompt(gemini_prompt)
                results['generated_response'] = response
            except Exception as e:
                logger.error(f"Gemini error: {str(e)}")
                results['generated_response'] = "Sorry, I encountered an error generating a response."
        else:
            results['generated_response'] = "I couldn't find any relevant information to answer your query."
        
        return jsonify(results)
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Check system health."""
        health = {
            'status': 'ok',
            'confluence': confluence_client.check_connection(),
            'remedy': remedy_client.check_connection(),
            'gemini': gemini_client.check_connection()
        }
        return jsonify(health)
    
    return app


# Create the Flask application
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
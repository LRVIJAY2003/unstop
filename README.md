import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # App settings
    DEBUG = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    SECRET_KEY = os.getenv('SECRET_KEY', 'enterprise-rag-default-secret-key')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Server settings
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '5000'))
    
    # Cache settings
    CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'True').lower() in ('true', '1', 't')
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_EXPIRATION', '3600'))  # 1 hour
    
    # RAG settings
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '768'))
    NUM_RESULTS = int(os.getenv('NUM_RESULTS', '5'))
    
    # Gemini settings
    PROJECT_ID = os.getenv('PROJECT_ID', 'prj-dv-cws-4363')
    REGION = os.getenv('REGION', 'us-central1')
    MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-2.0-flash-001')
    
    # Confluence settings
    CONFLUENCE_URL = os.getenv('CONFLUENCE_URL', '')
    CONFLUENCE_USERNAME = os.getenv('CONFLUENCE_USERNAME', '')
    CONFLUENCE_API_TOKEN = os.getenv('CONFLUENCE_API_TOKEN', '')
    
    # JIRA settings
    JIRA_URL = os.getenv('JIRA_URL', '')
    JIRA_USERNAME = os.getenv('JIRA_USERNAME', '')
    JIRA_API_TOKEN = os.getenv('JIRA_API_TOKEN', '')
    
    # Remedy settings
    REMEDY_URL = os.getenv('REMEDY_URL', 'https://cmegroup-restapi.onbmc.com')
    REMEDY_USERNAME = os.getenv('REMEDY_USERNAME', '')
    REMEDY_PASSWORD = os.getenv('REMEDY_PASSWORD', '')
    REMEDY_SSL_VERIFY = os.getenv('REMEDY_SSL_VERIFY', 'False').lower() in ('true', '1', 't')

    # Source settings
    AVAILABLE_SOURCES = ['confluence', 'jira', 'remedy']
    DEFAULT_SOURCES = os.getenv('DEFAULT_SOURCES', 'confluence,jira,remedy').split(',')
    
    # System prompts
    SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', """
    You are an enterprise assistant that helps employees find information from internal systems.
    You have access to Confluence for documentation, JIRA for tasks and issues, and Remedy for incidents.
    Provide clear, concise answers with specific references to where information was found.
    Always maintain a professional but friendly tone. Only respond based on the provided context.
    If you're unsure or the information isn't available, be honest about it.
    """)
    
    @classmethod
    def get_log_level(cls):
        """Convert string log level to logging level."""
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return levels.get(cls.LOG_LEVEL.upper(), logging.INFO)














# Application settings
DEBUG=False
SECRET_KEY=enterprise-rag-secret-key-change-in-production
LOG_LEVEL=INFO

# Server settings
HOST=0.0.0.0
PORT=5000

# Cache settings
CACHE_ENABLED=True
CACHE_EXPIRATION=3600

# RAG settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_DIMENSION=768
NUM_RESULTS=5

# Gemini settings
PROJECT_ID=prj-dv-cws-4363
REGION=us-central1
MODEL_NAME=gemini-2.0-flash-001

# Confluence settings
CONFLUENCE_URL=https://cmegroup.atlassian.net
CONFLUENCE_USERNAME=your-username
CONFLUENCE_API_TOKEN=your-api-token

# JIRA settings
JIRA_URL=https://your-jira-instance.atlassian.net
JIRA_USERNAME=your-username
JIRA_API_TOKEN=your-api-token

# Remedy settings
REMEDY_URL=https://cmegroup-restapi.onbmc.com
REMEDY_USERNAME=your-username
REMEDY_PASSWORD=your-password
REMEDY_SSL_VERIFY=False

# Source settings
DEFAULT_SOURCES=confluence,jira,remedy

# System prompt
SYSTEM_PROMPT=You are a professional enterprise assistant helping with questions about internal systems. You are knowledgeable about Confluence documentation, JIRA tasks and issues, and Remedy incidents. Provide concise, accurate responses with references to sources. Maintain a professional and helpful tone. When analyzing tables, diagrams, or structured data, provide clear insights. If information is incomplete or unavailable, be transparent about limitations.













import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from config import Config

def setup_logger():
    """
    Configure application logging with console and file handlers.
    """
    log_level = Config.get_log_level()
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates on reload
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/enterprise_rag.log',
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Reduce verbosity of some libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Log start message
    logger.info("Logger initialized")
    
    return logger











"""
Utility modules for the Enterprise RAG system.
"""






"""
Caching mechanisms for the RAG system.
"""
import logging
import json
import hashlib
from flask_caching import Cache
from functools import wraps
from config import Config

logger = logging.getLogger(__name__)

# Initialize cache
cache = Cache()

def init_cache(app):
    """
    Initialize the caching system with the Flask app.
    
    Args:
        app: Flask application instance
    """
    if Config.CACHE_ENABLED:
        cache_config = {
            'CACHE_TYPE': Config.CACHE_TYPE,
            'CACHE_DEFAULT_TIMEOUT': Config.CACHE_DEFAULT_TIMEOUT
        }
        cache.init_app(app, config=cache_config)
        logger.info(f"Cache initialized with type {Config.CACHE_TYPE}")
    else:
        logger.info("Caching is disabled")

def create_cache_key(*args, **kwargs):
    """
    Create a consistent cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        str: MD5 hash to use as cache key
    """
    # Convert all args and kwargs to a stable string representation
    key_parts = []
    
    # Add positional args
    for arg in args:
        try:
            if isinstance(arg, (dict, list, tuple, set)):
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                key_parts.append(str(arg))
        except (TypeError, ValueError):
            # If we can't serialize, use the string representation
            key_parts.append(str(arg))
    
    # Add keyword args (sorted by key)
    for key in sorted(kwargs.keys()):
        value = kwargs[key]
        try:
            if isinstance(value, (dict, list, tuple, set)):
                key_parts.append(f"{key}:{json.dumps(value, sort_keys=True)}")
            else:
                key_parts.append(f"{key}:{value}")
        except (TypeError, ValueError):
            key_parts.append(f"{key}:{str(value)}")
    
    # Join parts and create MD5 hash
    key_string = "::".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def cached(timeout=None):
    """
    Decorator for caching function results.
    
    Args:
        timeout: Cache timeout in seconds (None uses default)
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not Config.CACHE_ENABLED:
                return f(*args, **kwargs)
            
            # Create cache key from function args
            cache_key = f"{f.__module__}.{f.__name__}:{create_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return result
            
            # Not in cache, call function
            result = f(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, timeout=timeout)
            logger.debug(f"Cached result for {cache_key}")
            
            return result
        return wrapper
    return decorator

def invalidate_cache_prefix(prefix):
    """
    Invalidate all cache keys with given prefix.
    
    Args:
        prefix: Prefix of cache keys to invalidate
    """
    if not Config.CACHE_ENABLED or not hasattr(cache, 'cache'):
        return
    
    # Note: This works with SimpleCache but may not work with all cache types
    keys_to_delete = []
    for key in cache.cache.keys():
        if key.startswith(prefix):
            keys_to_delete.append(key)
    
    for key in keys_to_delete:
        cache.delete(key)
    
    logger.info(f"Invalidated {len(keys_to_delete)} cache keys with prefix '{prefix}'")














"""
Content parsing utilities for different document formats.
"""
import re
import logging
from typing import Dict, Any, List, Tuple
import html2text
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Initialize html2text converter with settings optimized for readability
h2t = html2text.HTML2Text()
h2t.ignore_links = False
h2t.ignore_images = False
h2t.ignore_tables = False
h2t.unicode_snob = True
h2t.body_width = 0  # Don't wrap lines
h2t.single_line_break = True  # Don't add extra line breaks


def extract_text_from_html(html_content: str) -> str:
    """
    Extract text from HTML content preserving structure.
    
    Args:
        html_content: HTML string
        
    Returns:
        Markdown formatted text
    """
    if not html_content:
        return ""
    
    try:
        # Convert HTML to Markdown using html2text
        markdown_text = h2t.handle(html_content)
        
        # Clean up extra whitespace
        markdown_text = re.sub(r'\n{3,}', '\n\n', markdown_text)
        
        return markdown_text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}")
        
        # Fallback to simple tag stripping
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator='\n\n').strip()


def extract_tables_from_html(html_content: str) -> List[Dict[str, Any]]:
    """
    Extract tables from HTML content.
    
    Args:
        html_content: HTML string
        
    Returns:
        List of table data with headers and rows
    """
    if not html_content:
        return []
    
    tables = []
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        html_tables = soup.find_all('table')
        
        for i, table in enumerate(html_tables):
            # Extract headers
            headers = []
            header_row = table.find('thead')
            
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all('th')]
            
            # If no headers found in thead, try first tr
            if not headers and table.find('tr'):
                headers = [th.get_text().strip() for th in table.find('tr').find_all(['th', 'td'])]
            
            # Extract rows
            rows = []
            body = table.find('tbody') or table
            
            # Skip header row if we used it for headers
            start_idx = 1 if not header_row and headers else 0
            
            for tr in body.find_all('tr')[start_idx:]:
                row = [td.get_text().strip() for td in tr.find_all(['td', 'th'])]
                if row:  # Skip empty rows
                    rows.append(row)
            
            # Create table data structure
            table_data = {
                'index': i,
                'headers': headers,
                'rows': rows,
                'num_rows': len(rows),
                'num_cols': len(headers) if headers else len(rows[0]) if rows else 0
            }
            
            tables.append(table_data)
        
        return tables
    except Exception as e:
        logger.error(f"Error extracting tables from HTML: {str(e)}")
        return []


def format_table_as_markdown(table: Dict[str, Any]) -> str:
    """
    Format table data as Markdown.
    
    Args:
        table: Table data dictionary
        
    Returns:
        Markdown formatted table
    """
    if not table or not table.get('rows'):
        return ""
    
    headers = table.get('headers', [])
    rows = table.get('rows', [])
    
    # If no headers but we have rows, create generic headers
    if not headers and rows:
        headers = [f"Column {i+1}" for i in range(len(rows[0]))]
    
    # Make sure all rows have same number of columns
    max_cols = max(len(headers), max(len(row) for row in rows))
    headers = headers + [''] * (max_cols - len(headers))
    rows = [row + [''] * (max_cols - len(row)) for row in rows]
    
    # Calculate column widths
    col_widths = []
    for i in range(max_cols):
        col_width = max(len(str(headers[i])), 
                        max(len(str(row[i])) for row in rows),
                        3)  # Minimum width of 3
        col_widths.append(min(col_width, 30))  # Cap width at 30
    
    # Build header row
    header_row = '| ' + ' | '.join(str(headers[i]).ljust(col_widths[i]) for i in range(max_cols)) + ' |'
    
    # Build separator row
    separator = '| ' + ' | '.join('-' * col_widths[i] for i in range(max_cols)) + ' |'
    
    # Build data rows
    data_rows = []
    for row in rows:
        data_row = '| ' + ' | '.join(str(row[i]).ljust(col_widths[i]) for i in range(max_cols)) + ' |'
        data_rows.append(data_row)
    
    # Combine all rows
    markdown_table = '\n'.join([header_row, separator] + data_rows)
    
    return markdown_table


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from Markdown text.
    
    Args:
        text: Markdown text
        
    Returns:
        List of code blocks with language and content
    """
    if not text:
        return []
    
    code_blocks = []
    
    # Match Markdown code blocks (```language\ncode```)
    pattern = r'```([a-zA-Z0-9_]*)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for i, (language, code) in enumerate(matches):
        language = language.strip() or 'text'
        code_blocks.append({
            'index': i,
            'language': language,
            'content': code.strip()
        })
    
    return code_blocks


def clean_text_for_embedding(text: str) -> str:
    """
    Clean and normalize text for embedding.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace URLs with placeholders
    text = re.sub(r'https?://\S+', '[URL]', text)
    
    # Replace excess whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,;:!?"\'-]', ' ', text)
    
    # Normalize whitespace again
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def split_text_by_headers(text: str) -> List[Tuple[str, str]]:
    """
    Split text into sections based on headers.
    
    Args:
        text: Markdown text
        
    Returns:
        List of (header, content) tuples
    """
    if not text:
        return []
    
    # Define regex pattern for Markdown headers
    header_pattern = r'^(#{1,6})\s+(.+)$'
    
    # Split the text into lines
    lines = text.split('\n')
    
    # Process each line
    sections = []
    current_header = ""
    current_content = []
    
    for line in lines:
        match = re.match(header_pattern, line)
        if match:
            # We found a header. Save the current section if it exists
            if current_header or current_content:
                sections.append((current_header, '\n'.join(current_content).strip()))
            
            # Start a new section
            current_header = match.group(2).strip()
            current_content = []
        else:
            # Add to current content
            current_content.append(line)
    
    # Add the last section
    if current_header or current_content:
        sections.append((current_header, '\n'.join(current_content).strip()))
    
    return sections


def summarize_text(text: str, max_length: int = 200) -> str:
    """
    Create a summary of text within max_length.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary
        
    Returns:
        Summarized text
    """
    if not text:
        return ""
    
    # If text is already shorter than max_length, return it
    if len(text) <= max_length:
        return text
    
    # Try to find a sentence break near max_length
    sentences = re.split(r'(?<=[.!?])\s+', text)
    summary = ""
    
    for sentence in sentences:
        if len(summary) + len(sentence) + 1 <= max_length:
            if summary:
                summary += " "
            summary += sentence
        else:
            break
    
    # If no good sentence break, just cut at max_length
    if not summary:
        summary = text[:max_length-3] + "..."
    
    return summary












"""
Retrieval-Augmented Generation (RAG) engine components.
"""















"""
Embedding utilities for the RAG engine using TF-IDF and SVD.
"""
import logging
import pickle
import os
import numpy as np
from typing import List, Dict, Any, Union, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from utils.cache import cached
from utils.content_parser import clean_text_for_embedding
from config import Config

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Handles text embedding using TF-IDF and SVD for dimensionality reduction.
    """
    
    def __init__(self, target_dimension: int = None):
        """
        Initialize the embedding engine.
        
        Args:
            target_dimension: Target dimension for embeddings
        """
        self.target_dimension = target_dimension or Config.EMBEDDING_DIMENSION
        self.vectorizer = None
        self.svd = None
        self.is_fitted = False
        self.actual_dimension = 0  # Track the actual dimension we can use
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Try to load pre-trained models
        self._load_models()
        
        logger.info(f"Initialized embedding engine with dimension={self.target_dimension}")
    
    def _load_models(self):
        """Load vectorizer and SVD models if they exist."""
        try:
            if os.path.exists('models/tfidf_vectorizer.pkl'):
                with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
                logger.info("Loaded TF-IDF vectorizer from file")
            
            if os.path.exists('models/svd_transformer.pkl'):
                with open('models/svd_transformer.pkl', 'rb') as f:
                    self.svd = pickle.load(f)
                logger.info("Loaded SVD transformer from file")
                
            if self.vectorizer and self.svd:
                self.is_fitted = True
                self.actual_dimension = self.svd.components_.shape[0]
                logger.info(f"Models loaded successfully. Actual dimension: {self.actual_dimension}")
        except Exception as e:
            logger.warning(f"Failed to load models: {str(e)}")
            self.vectorizer = None
            self.svd = None
            self.is_fitted = False
    
    def _save_models(self):
        """Save vectorizer and SVD models to disk."""
        try:
            if self.vectorizer:
                with open('models/tfidf_vectorizer.pkl', 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                
            if self.svd:
                with open('models/svd_transformer.pkl', 'wb') as f:
                    pickle.dump(self.svd, f)
                    
            logger.info("Saved models to disk")
        except Exception as e:
            logger.error(f"Failed to save models: {str(e)}")
    
    def _fit_models(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizer and SVD transformer.
        
        Args:
            texts: List of texts to fit the models on
        """
        if not texts:
            logger.warning("No texts provided for model fitting")
            return
        
        try:
            # Initialize and fit TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                analyzer='word',
                token_pattern=r'\w+',
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2)
            )
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Get number of features
            n_features = tfidf_matrix.shape[1]
            logger.debug(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            
            # Determine the appropriate SVD dimension
            self.actual_dimension = min(self.target_dimension, n_features - 1, len(texts) - 1)
            self.actual_dimension = max(self.actual_dimension, 1)  # Ensure at least 1 dimension
            
            logger.info(f"Using SVD dimension of {self.actual_dimension} (target was {self.target_dimension})")
            
            # Initialize and fit SVD transformer
            self.svd = TruncatedSVD(n_components=self.actual_dimension, random_state=42)
            self.svd.fit(tfidf_matrix)
            
            # Mark as fitted
            self.is_fitted = True
            
            # Save models
            self._save_models()
            
            logger.info(f"Models fitted successfully. Actual dimension: {self.actual_dimension}")
        except Exception as e:
            logger.error(f"Failed to fit models: {str(e)}")
            self.is_fitted = False
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text:
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.target_dimension)
        
        try:
            # Clean text
            text = clean_text_for_embedding(text)
            
            # Check if models are fitted
            if not self.is_fitted or not self.vectorizer or not self.svd:
                logger.warning("Models not fitted. Returning zero vector.")
                return np.zeros(self.target_dimension)
            
            # Transform text to TF-IDF vector
            tfidf_vector = self.vectorizer.transform([text])
            
            # Apply SVD
            svd_vector = self.svd.transform(tfidf_vector)[0]
            
            # Ensure consistent dimension
            if len(svd_vector) < self.target_dimension:
                padded_vector = np.zeros(self.target_dimension)
                padded_vector[:len(svd_vector)] = svd_vector
                return padded_vector
            else:
                return svd_vector[:self.target_dimension]
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            return np.zeros(self.target_dimension)
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        
        try:
            # Clean texts
            cleaned_texts = [clean_text_for_embedding(text) for text in texts]
            
            # Fit models if not fitted
            if not self.is_fitted or not self.vectorizer or not self.svd:
                logger.info("Models not fitted. Fitting now...")
                self._fit_models(cleaned_texts)
            
            # Check again if fitting was successful
            if not self.is_fitted:
                logger.warning("Model fitting failed. Returning zero vectors.")
                return [np.zeros(self.target_dimension) for _ in texts]
            
            # Transform texts to TF-IDF vectors
            tfidf_vectors = self.vectorizer.transform(cleaned_texts)
            
            # Apply SVD
            svd_vectors = self.svd.transform(tfidf_vectors)
            
            # Ensure consistent dimensions
            result_vectors = []
            for vec in svd_vectors:
                if len(vec) < self.target_dimension:
                    padded_vector = np.zeros(self.target_dimension)
                    padded_vector[:len(vec)] = vec
                    result_vectors.append(padded_vector)
                else:
                    result_vectors.append(vec[:self.target_dimension])
            
            return result_vectors
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            return [np.zeros(self.target_dimension) for _ in texts]
    
    def embed_document(self, document: Dict[str, Any]) -> np.ndarray:
        """
        Create embedding for a document.
        
        Args:
            document: Document with content and metadata
            
        Returns:
            Embedding vector
        """
        if not document or 'content' not in document:
            logger.warning("Invalid document provided for embedding")
            return np.zeros(self.target_dimension)
        
        content = document['content']
        return self.embed_text(content)
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Create embeddings for a list of documents.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of embedding vectors
        """
        if not documents:
            logger.warning("No documents provided for embedding")
            return []
        
        texts = []
        for doc in documents:
            if doc and 'content' in doc:
                texts.append(doc['content'])
            else:
                texts.append("")
        
        return self.embed_texts(texts)

# Global instance
embedding_engine = EmbeddingEngine()












"""
Document chunking strategies for the RAG engine.
"""
import re
import logging
from typing import List, Dict, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Handles document chunking with various strategies.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the chunker with configurable parameters.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        logger.debug(f"Initialized DocumentChunker with size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a document into chunks based on semantic boundaries.
        
        Args:
            document: Document with content and metadata
            
        Returns:
            List of document chunks with metadata
        """
        if not document or 'content' not in document:
            logger.warning("Empty document or missing content field")
            return []
        
        content = document['content']
        metadata = document.get('metadata', {})
        
        # Try to detect the document type and choose appropriate chunking strategy
        if self._has_semantic_structure(content):
            chunks = self._semantic_chunking(content)
        else:
            chunks = self._sliding_window_chunking(content)
        
        # Create chunk documents with metadata
        chunk_docs = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
                
            chunk_doc = {
                'content': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_index': i,
                    'chunk_total': len(chunks),
                    'source_document_id': metadata.get('id', 'unknown'),
                    'source_title': metadata.get('title', ''),
                    'source_type': metadata.get('type', ''),
                }
            }
            chunk_docs.append(chunk_doc)
        
        logger.debug(f"Chunked document into {len(chunk_docs)} parts")
        return chunk_docs
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        all_chunks = []
        for document in documents:
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def _has_semantic_structure(self, text: str) -> bool:
        """
        Detect if text has semantic structure like headers or paragraphs.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if semantic structure is detected
        """
        # Look for headers (markdown style) or multiple paragraphs
        header_pattern = r'^#{1,6}\s+.+$'
        has_headers = bool(re.search(header_pattern, text, re.MULTILINE))
        paragraph_breaks = text.count('\n\n')
        
        return has_headers or paragraph_breaks > 3
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """
        Split text by semantic boundaries like headers or paragraphs.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        # Split by headers (markdown style)
        header_pattern = r'^(#{1,6}\s+.+)$'
        parts = re.split(header_pattern, text, flags=re.MULTILINE)
        
        # If we found headers
        if len(parts) > 1:
            chunks = []
            current_chunk = ""
            current_size = 0
            
            # Process each part (alternating between header and content)
            for i in range(0, len(parts)):
                part = parts[i].strip()
                if not part:
                    continue
                    
                # If this part is a header or the current chunk is getting too big
                if re.match(header_pattern, part) or current_size + len(part) > self.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = part
                    current_size = len(part)
                else:
                    current_chunk += "\n\n" + part
                    current_size += len(part)
            
            # Add the last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            # If we still have large chunks, break them down further
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > self.chunk_size:
                    final_chunks.extend(self._sliding_window_chunking(chunk))
                else:
                    final_chunks.append(chunk)
                    
            return final_chunks
        
        # Fall back to paragraph splitting if no headers found
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph would exceed chunk size
            if current_size + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
                current_size = len(para)
            else:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
                current_size += len(para) + 2  # +2 for the newlines
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
            
        # If we have paragraphs that are too large, break them down
        if any(len(chunk) > self.chunk_size for chunk in chunks):
            return self._sliding_window_chunking(text)
            
        return chunks
        
    def _sliding_window_chunking(self, text: str) -> List[str]:
        """
        Split text using sliding window approach with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        # Special case for short text
        if len(text) <= self.chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end position
            end = start + self.chunk_size
            
            # Adjust end to avoid cutting words
            if end < len(text):
                # Try to find sentence boundary
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Try to find word boundary
                    space = text.rfind(' ', start, end)
                    if space > start + self.chunk_size // 2:
                        end = space
            else:
                end = len(text)
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position for next chunk, accounting for overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)
        
        return chunks

# Global instance
document_chunker = DocumentChunker()















"""
Vector search and hybrid retrieval for the RAG engine.
"""
import logging
import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from rag_engine.embedding import embedding_engine
from config import Config

logger = logging.getLogger(__name__)

class RetrievalEngine:
    """
    Handles document retrieval using vector search and BM25.
    """
    
    def __init__(self, vector_dimension: int = None):
        """
        Initialize the retrieval engine.
        
        Args:
            vector_dimension: Dimension of the embedding vectors
        """
        self.vector_dimension = vector_dimension or Config.EMBEDDING_DIMENSION
        self.index = None
        self.documents = []
        self.bm25_vectorizer = None
        self.is_initialized = False
        
        # Directory for storing indexes
        os.makedirs('indexes', exist_ok=True)
        
        logger.info(f"Initialized retrieval engine with dimension={self.vector_dimension}")
    
    def initialize(self, documents: List[Dict[str, Any]], force_rebuild: bool = False):
        """
        Initialize retrieval engine with documents.
        
        Args:
            documents: List of documents to index
            force_rebuild: Force rebuild index even if it exists
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return
        
        # Try to load existing index and documents
        if not force_rebuild and os.path.exists('indexes/document_index.faiss'):
            try:
                # Load documents
                with open('indexes/documents.pkl', 'rb') as f:
                    self.documents = pickle.load(f)
                
                # Load FAISS index
                self.index = faiss.read_index('indexes/document_index.faiss')
                
                # Load BM25 vectorizer
                if os.path.exists('indexes/bm25_vectorizer.pkl'):
                    with open('indexes/bm25_vectorizer.pkl', 'rb') as f:
                        self.bm25_vectorizer = pickle.load(f)
                
                self.is_initialized = True
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
                return
            except Exception as e:
                logger.error(f"Failed to load existing index: {str(e)}")
        
        # Build from scratch
        self._build_index(documents)
    
    def _build_index(self, documents: List[Dict[str, Any]]):
        """
        Build search index from documents.
        
        Args:
            documents: List of documents to index
        """
        try:
            logger.info(f"Building index with {len(documents)} documents")
            
            # Store documents
            self.documents = documents
            
            # Extract text for BM25 indexing
            document_texts = [doc.get('content', '') for doc in documents]
            
            # Build BM25 index (using TfidfVectorizer with BM25 parameters)
            self.bm25_vectorizer = TfidfVectorizer(
                analyzer='word',
                token_pattern=r'\w+',
                min_df=1,
                max_df=0.95,
                sublinear_tf=True  # Apply sublinear tf scaling (BM25-like)
            )
            self.bm25_vectorizer.fit(document_texts)
            
            # Create embeddings for vector search
            document_embeddings = embedding_engine.embed_documents(documents)
            
            # Convert to numpy array
            embeddings_array = np.array(document_embeddings).astype('float32')
            
            # Create FAISS index
            self.index = faiss.IndexFlatL2(self.vector_dimension)
            if len(documents) > 0:
                self.index.add(embeddings_array)
            
            # Save index and documents
            self._save_index()
            
            self.is_initialized = True
            logger.info(f"Index built successfully with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to build index: {str(e)}")
            self.is_initialized = False
    
    def _save_index(self):
        """Save index and documents to disk."""
        try:
            # Save documents
            with open('indexes/documents.pkl', 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save FAISS index
            if self.index:
                faiss.write_index(self.index, 'indexes/document_index.faiss')
            
            # Save BM25 vectorizer
            if self.bm25_vectorizer:
                with open('indexes/bm25_vectorizer.pkl', 'wb') as f:
                    pickle.dump(self.bm25_vectorizer, f)
            
            logger.info("Saved index to disk")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add new documents to the index.
        
        Args:
            documents: List of documents to add
        """
        if not documents:
            return
        
        if not self.is_initialized:
            self.initialize(documents)
            return
        
        try:
            # Create embeddings for new documents
            document_embeddings = embedding_engine.embed_documents(documents)
            
            # Convert to numpy array
            embeddings_array = np.array(document_embeddings).astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Add to documents list
            self.documents.extend(documents)
            
            # Extract text for BM25 reindexing
            all_document_texts = [doc.get('content', '') for doc in self.documents]
            
            # Rebuild BM25 index
            self.bm25_vectorizer = TfidfVectorizer(
                analyzer='word',
                token_pattern=r'\w+',
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
            )
            self.bm25_vectorizer.fit(all_document_texts)
            
            # Save updated index and documents
            self._save_index()
            
            logger.info(f"Added {len(documents)} documents to index")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query using hybrid retrieval (vector + BM25).
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        top_k = top_k or Config.NUM_RESULTS
        
        if not self.is_initialized:
            logger.warning("Retrieval engine not initialized")
            return []
        
        try:
            # Get vector search results
            vector_results = self._vector_search(query, top_k=top_k*2)
            
            # Get BM25 results
            bm25_results = self._bm25_search(query, top_k=top_k*2)
            
            # Merge results (simple score normalization and weighted sum)
            merged_results = self._hybrid_merge(vector_results, bm25_results, top_k)
            
            logger.info(f"Retrieved {len(merged_results)} documents for query: {query}")
            return merged_results
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return []
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        # Create query embedding
        query_embedding = embedding_engine.embed_text(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search index
        D, I = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0 or idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            score = 1.0 / (1.0 + dist)  # Convert distance to similarity score
            
            result = {
                'document': doc,
                'score': float(score),
                'rank': i + 1,
                'method': 'vector'
            }
            results.append(result)
        
        return results
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform BM25-like keyword search.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents with scores
        """
        if not self.bm25_vectorizer:
            return []
        
        # Transform query to vector
        query_vector = self.bm25_vectorizer.transform([query])
        
        # Calculate scores for all documents
        document_texts = [doc.get('content', '') for doc in self.documents]
        doc_vectors = self.bm25_vectorizer.transform(document_texts)
        
        # Calculate dot product for similarity
        scores = (query_vector * doc_vectors.T).toarray()[0]
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Format results
        results = []
        for i, idx in enumerate(top_indices):
            score = scores[idx]
            if score <= 0:
                continue
                
            doc = self.documents[idx]
            
            result = {
                'document': doc,
                'score': float(score),
                'rank': i + 1,
                'method': 'bm25'
            }
            results.append(result)
        
        return results
    
    def _hybrid_merge(self, vector_results: List[Dict[str, Any]], 
                      bm25_results: List[Dict[str, Any]], 
                      top_k: int) -> List[Dict[str, Any]]:
        """
        Merge results from vector and BM25 search.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            top_k: Number of final results to return
            
        Returns:
            Merged list of results
        """
        # Normalize scores within each method
        if vector_results:
            max_vector_score = max(r['score'] for r in vector_results)
            for r in vector_results:
                r['normalized_score'] = r['score'] / max_vector_score if max_vector_score > 0 else 0
        
        if bm25_results:
            max_bm25_score = max(r['score'] for r in bm25_results)
            for r in bm25_results:
                r['normalized_score'] = r['score'] / max_bm25_score if max_bm25_score > 0 else 0
        
        # Combine results
        doc_map = {}
        
        # Add vector results
        for r in vector_results:
            doc_id = r['document'].get('metadata', {}).get('id', id(r['document']))
            doc_map[doc_id] = {
                'document': r['document'],
                'vector_score': r['normalized_score'],
                'bm25_score': 0.0,
                'method': 'vector'
            }
        
        # Add/merge BM25 results
        for r in bm25_results:
            doc_id = r['document'].get('metadata', {}).get('id', id(r['document']))
            if doc_id in doc_map:
                doc_map[doc_id]['bm25_score'] = r['normalized_score']
                doc_map[doc_id]['method'] = 'hybrid'
            else:
                doc_map[doc_id] = {
                    'document': r['document'],
                    'vector_score': 0.0,
                    'bm25_score': r['normalized_score'],
                    'method': 'bm25'
                }
        
        # Calculate final scores (weighted sum)
        vector_weight = 0.7  # Weight for vector score
        bm25_weight = 0.3    # Weight for BM25 score
        
        results = []
        for doc_id, data in doc_map.items():
            final_score = (data['vector_score'] * vector_weight + 
                          data['bm25_score'] * bm25_weight)
            
            result = {
                'document': data['document'],
                'score': final_score,
                'method': data['method'],
                'vector_score': data['vector_score'],
                'bm25_score': data['bm25_score']
            }
            results.append(result)
        
        # Sort by final score and limit to top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

# Global instance
retrieval_engine = RetrievalEngine()













"""
Google Gemini API integration for the RAG engine.
"""
import logging
import time
from typing import List, Dict, Any, Optional, Generator
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from google.api_core.exceptions import GoogleAPICallError
from utils.cache import cached
from config import Config

logger = logging.getLogger(__name__)

class GeminiClient:
    """
    Client for interacting with Google's Gemini models via Vertex AI.
    """
    
    def __init__(self, project_id: str = None, region: str = None, model_name: str = None):
        """
        Initialize the Gemini client.
        
        Args:
            project_id: Google Cloud project ID
            region: Google Cloud region
            model_name: Gemini model name
        """
        self.project_id = project_id or Config.PROJECT_ID
        self.region = region or Config.REGION
        self.model_name = model_name or Config.MODEL_NAME
        self._model = None
        self._initialized = False
        
        # Initialize Vertex AI
        self._initialize_vertex_ai()
        
        logger.info(f"Initialized Gemini client with model={self.model_name}")
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI client."""
        try:
            vertexai.init(project=self.project_id, location=self.region)
            self._model = GenerativeModel(self.model_name)
            self._initialized = True
            logger.info(f"Vertex AI initialized with project={self.project_id}, region={self.region}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {str(e)}")
            self._initialized = False
    
    def generate_response(self, prompt: str, system_prompt: str = None, 
                         temperature: float = 0.7, max_output_tokens: int = 8192) -> str:
        """
        Generate a response using the Gemini model.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions for the model
            temperature: Temperature for generation
            max_output_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response text
        """
        if not self._initialized:
            logger.error("Gemini client not properly initialized")
            return "Sorry, I'm having trouble connecting to the AI service. Please try again later."
        
        try:
            # Prepare the prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_output_tokens,
            )
            
            # Generate response
            logger.info(f"Generating response with temperature={temperature}")
            response = self._model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )
            
            # Extract text from response
            if response and hasattr(response, 'text'):
                return response.text
            else:
                logger.warning(f"Empty or invalid response: {response}")
                return "I couldn't generate a response. Please try rephrasing your question."
        except GoogleAPICallError as e:
            logger.error(f"Gemini API error: {str(e)}")
            return f"There was an error communicating with the AI service: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in generate_response: {str(e)}")
            return "Sorry, an unexpected error occurred. Please try again later."
    
    def generate_stream(self, prompt: str, system_prompt: str = None,
                       temperature: float = 0.7, max_output_tokens: int = 8192) -> Generator[str, None, None]:
        """
        Generate a streaming response using the Gemini model.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions for the model
            temperature: Temperature for generation
            max_output_tokens: Maximum number of tokens to generate
            
        Yields:
            Chunks of generated text
        """
        if not self._initialized:
            logger.error("Gemini client not properly initialized")
            yield "Sorry, I'm having trouble connecting to the AI service. Please try again later."
            return
        
        try:
            # Prepare the prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_output_tokens,
            )
            
            # Generate streaming response
            logger.info(f"Generating streaming response with temperature={temperature}")
            response_stream = self._model.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            )
            
            # Stream the response chunks
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
        except GoogleAPICallError as e:
            logger.error(f"Gemini API error in streaming: {str(e)}")
            yield f"\nThere was an error communicating with the AI service: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in generate_stream: {str(e)}")
            yield "\nSorry, an unexpected error occurred. Please try again later."
    
    def generate_with_sources(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                             temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a response with source attribution.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents to use as context
            temperature: Temperature for generation
            
        Returns:
            Dictionary with response and sources
        """
        if not retrieved_docs:
            logger.warning("No documents provided for context")
            return {
                "response": "I don't have enough information to answer that question accurately.",
                "sources": []
            }
        
        # Prepare context from retrieved documents
        context_parts = []
        sources = []
        
        for i, result in enumerate(retrieved_docs):
            doc = result.get('document', {})
            content = doc.get('content', '').strip()
            metadata = doc.get('metadata', {})
            
            if not content:
                continue
                
            # Add to context
            context_parts.append(f"[Document {i+1}]\n{content}\n")
            
            # Add to sources
            source = {
                "title": metadata.get('title', f"Document {i+1}"),
                "url": metadata.get('url', ''),
                "type": metadata.get('type', ''),
                "id": metadata.get('id', ''),
                "score": result.get('score', 0.0)
            }
            sources.append(source)
        
        context = "\n".join(context_parts)
        
        # Create the prompt with context and query
        system_instructions = Config.SYSTEM_PROMPT + """
        Answer based ONLY on the information provided in the documents.
        If the answer cannot be determined from the documents, say so clearly.
        Reference relevant document numbers in your answer.
        """
        
        prompt = f"""
        Documents:
        {context}
        
        User question: {query}
        
        Please answer the question based only on the provided documents. Include references to the document numbers where you found the information.
        """
        
        # Generate response
        response_text = self.generate_response(
            prompt=prompt,
            system_prompt=system_instructions,
            temperature=temperature
        )
        
        return {
            "response": response_text,
            "sources": sources
        }

# Global instance
gemini_client = GeminiClient()















"""
Main RAG processor integrating all components.
"""
import logging
from typing import List, Dict, Any, Optional, Generator, Tuple
from concurrent.futures import ThreadPoolExecutor
from rag_engine.chunking import document_chunker
from rag_engine.embedding import embedding_engine
from rag_engine.retrieval import retrieval_engine
from rag_engine.gemini_integration import gemini_client
from utils.cache import cached
from config import Config

logger = logging.getLogger(__name__)

class RAGProcessor:
    """
    Main processor for the RAG system.
    """
    
    def __init__(self):
        """Initialize the RAG processor."""
        logger.info("Initializing RAG processor")
    
    def process_documents(self, documents: List[Dict[str, Any]], force_rebuild: bool = False) -> bool:
        """
        Process documents for the RAG system.
        
        Args:
            documents: List of documents to process
            force_rebuild: Force rebuild of indexes
            
        Returns:
            Success flag
        """
        try:
            # Chunk documents
            logger.info(f"Chunking {len(documents)} documents")
            chunked_documents = document_chunker.chunk_documents(documents)
            
            # Initialize retrieval engine with chunks
            logger.info(f"Indexing {len(chunked_documents)} document chunks")
            if force_rebuild:
                retrieval_engine.initialize(chunked_documents, force_rebuild=True)
            else:
                retrieval_engine.add_documents(chunked_documents)
            
            logger.info("Document processing completed successfully")
            return True
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            return False
    
    def process_query(self, query: str, sources: List[str] = None, 
                     top_k: int = None) -> Dict[str, Any]:
        """
        Process a user query and generate response.
        
        Args:
            query: User query
            sources: List of data sources to use
            top_k: Number of documents to retrieve
            
        Returns:
            Response with context and sources
        """
        try:
            # Use default sources if not specified
            sources = sources or Config.DEFAULT_SOURCES
            top_k = top_k or Config.NUM_RESULTS
            
            logger.info(f"Processing query: '{query}' with sources={sources}")
            
            # Retrieve relevant documents
            relevant_docs = retrieval_engine.retrieve(query, top_k=top_k)
            
            # If no results, return basic response
            if not relevant_docs:
                logger.warning(f"No relevant documents found for query: {query}")
                return {
                    "query": query,
                    "response": "I don't have enough information to answer that question accurately.",
                    "sources": [],
                    "success": True
                }
            
            # Generate response with context
            logger.info(f"Generating response with {len(relevant_docs)} documents")
            result = gemini_client.generate_with_sources(query, relevant_docs)
            
            # Add query to result
            result["query"] = query
            result["success"] = True
            
            logger.info(f"Query processed successfully")
            return result
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "query": query,
                "response": f"Sorry, an error occurred while processing your query: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def process_query_stream(self, query: str, sources: List[str] = None, 
                            top_k: int = None) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """
        Process a user query and generate a streaming response.
        
        Args:
            query: User query
            sources: List of data sources to use
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (response generator, sources list)
        """
        try:
            # Use default sources if not specified
            sources = sources or Config.DEFAULT_SOURCES
            top_k = top_k or Config.NUM_RESULTS
            
            logger.info(f"Processing streaming query: '{query}' with sources={sources}")
            
            # Retrieve relevant documents
            relevant_docs = retrieval_engine.retrieve(query, top_k=top_k)
            
            # Extract sources
            sources_list = []
            for result in relevant_docs:
                doc = result.get('document', {})
                metadata = doc.get('metadata', {})
                source = {
                    "title": metadata.get('title', 'Unknown'),
                    "url": metadata.get('url', ''),
                    "type": metadata.get('type', ''),
                    "id": metadata.get('id', ''),
                    "score": result.get('score', 0.0)
                }
                sources_list.append(source)
            
            # If no results, return basic response
            if not relevant_docs:
                logger.warning(f"No relevant documents found for query: {query}")
                def empty_generator():
                    yield "I don't have enough information to answer that question accurately."
                return empty_generator(), []
            
            # Prepare context from retrieved documents
            context_parts = []
            
            for i, result in enumerate(relevant_docs):
                doc = result.get('document', {})
                content = doc.get('content', '').strip()
                
                if not content:
                    continue
                    
                # Add to context
                context_parts.append(f"[Document {i+1}]\n{content}\n")
            
            context = "\n".join(context_parts)
            
            # Create the prompt with context and query
            system_instructions = Config.SYSTEM_PROMPT + """
            Answer based ONLY on the information provided in the documents.
            If the answer cannot be determined from the documents, say so clearly.
            Reference relevant document numbers in your answer.
            """
            
            prompt = f"""
            Documents:
            {context}
            
            User question: {query}
            
            Please answer the question based only on the provided documents. Include references to the document numbers where you found the information.
            """
            
            # Generate streaming response
            logger.info(f"Generating streaming response with {len(relevant_docs)} documents")
            response_stream = gemini_client.generate_stream(
                prompt=prompt,
                system_prompt=system_instructions
            )
            
            logger.info(f"Streaming query processed successfully")
            return response_stream, sources_list
        except Exception as e:
            logger.error(f"Streaming query processing failed: {str(e)}")
            def error_generator():
                yield f"Sorry, an error occurred while processing your query: {str(e)}"
            return error_generator(), []

# Global instance
rag_processor = RAGProcessor()












"""
Data source connectors for the RAG system.
"""












"""
Confluence integration module.
"""










"""
JIRA integration module.
"""










"""
Remedy integration module.
"""












"""
Confluence API client for the RAG system.
"""
import logging
import requests
import json
import urllib3
from typing import Dict, List, Any, Optional
from html.parser import HTMLParser
from utils.content_parser import extract_text_from_html
from utils.cache import cached
from config import Config

logger = logging.getLogger(__name__)

class HTMLFilter(HTMLParser):
    """
    Filter HTML content to extract text.
    """
    def __init__(self):
        super().__init__()
        self.text = ""
    
    def handle_data(self, data):
        self.text += data + " "

class ConfluenceClient:
    """
    Client for Confluence REST API operations with comprehensive error handling.
    """
    
    def __init__(self, base_url: str = None, username: str = None, api_token: str = None, ssl_verify: bool = True):
        """
        Initialize the Confluence client with server and authentication details.
        
        Args:
            base_url: The base URL of the Confluence server
            username: Username for authentication
            api_token: API token for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.base_url = base_url or Config.CONFLUENCE_URL
        self.base_url = self.base_url.rstrip('/')
        self.username = username or Config.CONFLUENCE_USERNAME
        self.api_token = api_token or Config.CONFLUENCE_API_TOKEN
        
        # Handle SSL verification
        if not ssl_verify:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled.")
        self.ssl_verify = ssl_verify
        
        # Set up headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized Confluence client for {self.base_url}")
    
    def test_connection(self):
        """
        Test the connection to Confluence API.
        
        Returns:
            tuple: (success, server_version)
        """
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.base_url}/rest/api/serverInfo",
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            server_info = response.json()
            server_version = server_info.get('version', 'Unknown')
            
            logger.info(f"Connection to Confluence successful! Server version: {server_version}")
            return True, server_version
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except ValueError:
                    logger.error(f"Response content: {e.response.text}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Connection test failed with unexpected error: {str(e)}")
            return False, str(e)
    
    @cached(timeout=3600)  # Cache for 1 hour
    def get_content_by_id(self, content_id: str, expand: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific content by its ID.
        
        Args:
            content_id: The ID of the content to retrieve
            expand: Optional list of properties to expand
            
        Returns:
            dict: Content data or None if not found/error
        """
        try:
            # Build expand parameter
            expand_param = ",".join(expand) if expand else "body.storage,metadata.labels"
            
            logger.info(f"Fetching content: {content_id}")
            response = requests.get(
                f"{self.base_url}/rest/api/content/{content_id}",
                auth=(self.username, self.api_token),
                headers=self.headers,
                params={"expand": expand_param},
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                content = response.json()
                logger.info(f"Successfully retrieved content: {content_id}")
                return content
            else:
                logger.error(f"Failed to get content {content_id}: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting content {content_id}: {str(e)}")
            return None
    
    @cached(timeout=3600)
    def search_content(self, cql: str = None, title: str = None, content_type: str = "page", 
                       expand: Optional[List[str]] = None, limit: int = 25, start: int = 0) -> List[Dict[str, Any]]:
        """
        Search for content using CQL or specific parameters.
        
        Args:
            cql: Confluence Query Language string
            title: Title to search for
            content_type: Type of content to search for (default: page)
            expand: Optional list of properties to expand
            limit: Maximum number of results to return
            start: Starting index for pagination
            
        Returns:
            list: List of content items
        """
        try:
            params = {
                "limit": limit,
                "start": start
            }
            
            # Build CQL if not provided
            if not cql:
                query_parts = []
                
                if content_type:
                    query_parts.append(f"type={content_type}")
                
                if title:
                    # Escape special characters in title
                    safe_title = title.replace('"', '\\"')
                    query_parts.append(f'title~"{safe_title}"')
                
                if query_parts:
                    params["cql"] = " AND ".join(query_parts)
            else:
                params["cql"] = cql
            
            # Add expand parameter if provided
            if expand:
                params["expand"] = ",".join(expand)
            
            logger.info(f"Searching for content with params: {params}")
            response = requests.get(
                f"{self.base_url}/rest/api/content/search",
                auth=(self.username, self.api_token),
                headers=self.headers,
                params=params,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                results = response.json()
                logger.info(f"Search returned {len(results.get('results', []))} results")
                return results.get('results', [])
            else:
                logger.error(f"Failed to search content: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            return []
    
    @cached(timeout=7200)  # Cache for 2 hours
    def get_all_content(self, content_type: str = "page", limit: int = 25) -> List[Dict[str, Any]]:
        """
        Get all content of specified type with pagination handling.
        
        Args:
            content_type: Type of content to retrieve
            limit: Maximum number of results per request
            
        Returns:
            list: List of all content items
        """
        all_content = []
        start = 0
        
        logger.info(f"Retrieving all {content_type} content")
        
        while True:
            try:
                params = {
                    "type": content_type,
                    "limit": limit,
                    "start": start
                }
                
                response = requests.get(
                    f"{self.base_url}/rest/api/content",
                    auth=(self.username, self.api_token),
                    headers=self.headers,
                    params=params,
                    verify=self.ssl_verify
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    all_content.extend(results)
                    
                    # Check if there are more pages
                    if len(results) < limit:
                        break
                    
                    # Move to next page
                    start += limit
                    logger.info(f"Retrieved {len(all_content)} {content_type} items so far")
                else:
                    logger.error(f"Failed to retrieve all content: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    break
            except Exception as e:
                logger.error(f"Error retrieving all content: {str(e)}")
                break
        
        logger.info(f"Retrieved a total of {len(all_content)} {content_type} items")
        return all_content
    
    @cached(timeout=3600)
    def get_space(self, space_key: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific Confluence space.
        
        Args:
            space_key: The key of the space
            
        Returns:
            dict: Space data or None if not found/error
        """
        try:
            logger.info(f"Fetching space: {space_key}")
            response = requests.get(
                f"{self.base_url}/rest/api/space/{space_key}",
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                space = response.json()
                logger.info(f"Successfully retrieved space: {space_key}")
                return space
            else:
                logger.error(f"Failed to get space {space_key}: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting space {space_key}: {str(e)}")
            return None
    
    @cached(timeout=7200)  # Cache for 2 hours
    def get_all_spaces(self, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Get all spaces with pagination handling.
        
        Args:
            limit: Maximum number of results per request
            
        Returns:
            list: List of all spaces
        """
        all_spaces = []
        start = 0
        
        logger.info("Retrieving all spaces")
        
        while True:
            try:
                params = {
                    "limit": limit,
                    "start": start
                }
                
                response = requests.get(
                    f"{self.base_url}/rest/api/space",
                    auth=(self.username, self.api_token),
                    headers=self.headers,
                    params=params,
                    verify=self.ssl_verify
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    all_spaces.extend(results)
                    
                    # Check if there are more pages
                    if len(results) < limit:
                        break
                    
                    # Move to next page
                    start += limit
                    logger.info(f"Retrieved {len(all_spaces)} spaces so far")
                else:
                    logger.error(f"Failed to retrieve all spaces: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    break
            except Exception as e:
                logger.error(f"Error retrieving all spaces: {str(e)}")
                break
        
        logger.info(f"Retrieved a total of {len(all_spaces)} spaces")
        return all_spaces
    
    def get_page_content(self, page_id: str, expand: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get the content of a page in a suitable format for RAG.
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            page_id: The ID of the page
            expand: Optional list of properties to expand
            
        Returns:
            dict: Processed page content or None
        """
        try:
            # Get page with body.storage and metadata
            expand = expand or ["body.storage", "metadata.labels"]
            page = self.get_content_by_id(page_id, expand=expand)
            
            if not page:
                return None
            
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/wiki{page.get('_links', {}).get('webui', '')}",
                "created": page.get("created"),
                "updated": page.get("history", {}).get("modified"),
                "labels": [label.get("name") for label in page.get("metadata", {}).get("labels", {}).get("results", [])]
            }
            
            # Get the content
            body = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Convert HTML to plain text
            plain_text = extract_text_from_html(body)
            
            # Return structured content
            return {
                "metadata": metadata,
                "content": plain_text,
                "raw_html": body
            }
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def get_all_content_for_rag(self, content_type: str = "page", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve all content and process it for RAG.
        
        Args:
            content_type: Type of content to retrieve
            limit: Maximum number of results for initial query
            
        Returns:
            list: Processed content items ready for RAG
        """
        try:
            # Get all content IDs first
            contents = self.get_all_content(content_type=content_type, limit=limit)
            content_ids = [content.get("id") for content in contents]
            
            # Process each content item
            processed_contents = []
            
            for content_id in content_ids:
                processed_content = self.get_page_content(content_id)
                if processed_content:
                    processed_contents.append(processed_content)
            
            logger.info(f"Processed {len(processed_contents)} content items for RAG")
            return processed_contents
        except Exception as e:
            logger.error(f"Error retrieving content for RAG: {str(e)}")
            return []

# Create global instance
confluence_client = ConfluenceClient()












"""
Confluence connector for the RAG system.
"""
import logging
from typing import List, Dict, Any, Optional
from data_sources.confluence.client import confluence_client
from utils.cache import cached

logger = logging.getLogger(__name__)

class ConfluenceConnector:
    """
    Connector between Confluence and the RAG system.
    """
    
    def __init__(self):
        """Initialize the Confluence connector."""
        self.client = confluence_client
        logger.info("Initialized Confluence connector")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Confluence.
        
        Returns:
            Dict with connection status and details
        """
        success, details = self.client.test_connection()
        return {
            "success": success,
            "details": details,
            "source": "confluence"
        }
    
    @cached(timeout=3600)  # Cache for 1 hour
    def get_spaces(self) -> List[Dict[str, Any]]:
        """
        Get all Confluence spaces.
        
        Returns:
            List of spaces with key information
        """
        try:
            spaces = self.client.get_all_spaces()
            # Extract relevant information
            space_info = []
            for space in spaces:
                space_info.append({
                    "id": space.get("id"),
                    "key": space.get("key"),
                    "name": space.get("name"),
                    "type": space.get("type"),
                    "description": space.get("description", {}).get("plain", {}).get("value", "")
                })
            return space_info
        except Exception as e:
            logger.error(f"Error getting spaces: {str(e)}")
            return []
    
    @cached(timeout=3600)  # Cache for 1 hour
    def search_pages(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for pages in Confluence.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching pages
        """
        try:
            # Use CQL for text search
            cql = f'type=page AND text ~ "{query}"'
            results = self.client.search_content(
                cql=cql,
                content_type="page",
                limit=limit
            )
            
            # Format results
            pages = []
            for result in results:
                pages.append({
                    "id": result.get("id"),
                    "title": result.get("title"),
                    "type": result.get("type"),
                    "url": f"{self.client.base_url}/wiki{result.get('_links', {}).get('webui', '')}"
                })
            
            return pages
        except Exception as e:
            logger.error(f"Error searching pages: {str(e)}")
            return []
    
    def get_page_for_rag(self, page_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single page for RAG processing.
        
        Args:
            page_id: ID of the page
            
        Returns:
            Processed page content or None
        """
        try:
            page_content = self.client.get_page_content(page_id)
            if not page_content:
                return None
            
            # Format for RAG
            return {
                "content": page_content.get("content", ""),
                "metadata": page_content.get("metadata", {})
            }
        except Exception as e:
            logger.error(f"Error getting page for RAG: {str(e)}")
            return None
    
    def get_all_pages_for_rag(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all pages for RAG processing.
        
        Args:
            limit: Maximum number of pages to process
            
        Returns:
            List of processed pages
        """
        try:
            page_contents = self.client.get_all_content_for_rag(limit=limit)
            
            # Format for RAG
            rag_documents = []
            for page in page_contents:
                if not page or "content" not in page:
                    continue
                
                rag_document = {
                    "content": page.get("content", ""),
                    "metadata": page.get("metadata", {})
                }
                rag_documents.append(rag_document)
            
            logger.info(f"Retrieved {len(rag_documents)} Confluence pages for RAG")
            return rag_documents
        except Exception as e:
            logger.error(f"Error getting pages for RAG: {str(e)}")
            return []
    
    def get_space_pages_for_rag(self, space_key: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all pages in a space for RAG processing.
        
        Args:
            space_key: Key of the space
            limit: Maximum number of pages to process
            
        Returns:
            List of processed pages
        """
        try:
            # Search for pages in the space
            cql = f'type=page AND space="{space_key}"'
            results = self.client.search_content(
                cql=cql,
                content_type="page",
                limit=limit
            )
            
            # Process each page
            rag_documents = []
            for result in results:
                page_id = result.get("id")
                page = self.get_page_for_rag(page_id)
                if page:
                    rag_documents.append(page)
            
            logger.info(f"Retrieved {len(rag_documents)} pages from space {space_key} for RAG")
            return rag_documents
        except Exception as e:
            logger.error(f"Error getting space pages for RAG: {str(e)}")
            return []

# Global instance
confluence_connector = ConfluenceConnector()















"""
JIRA API client for the RAG system.
"""
import logging
import requests
import json
import urllib3
from typing import Dict, List, Any, Optional, Tuple
from utils.content_parser import extract_text_from_html
from utils.cache import cached
from config import Config

logger = logging.getLogger(__name__)

class JIRAClient:
    """
    Client for JIRA REST API operations with comprehensive error handling.
    """
    
    def __init__(self, base_url: str = None, username: str = None, api_token: str = None, ssl_verify: bool = True):
        """
        Initialize the JIRA client with server and authentication details.
        
        Args:
            base_url: The base URL of the JIRA server
            username: Username for authentication
            api_token: API token for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.base_url = base_url or Config.JIRA_URL
        self.base_url = self.base_url.rstrip('/')
        self.username = username or Config.JIRA_USERNAME
        self.api_token = api_token or Config.JIRA_API_TOKEN
        
        # Handle SSL verification
        if not ssl_verify:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled.")
        self.ssl_verify = ssl_verify
        
        # Set up headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized JIRA client for {self.base_url}")
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the connection to JIRA API.
        
        Returns:
            tuple: (success, server_info)
        """
        try:
            logger.info("Testing connection to JIRA...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/serverInfo",
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            server_info = response.json()
            server_version = server_info.get('version', 'Unknown')
            
            logger.info(f"Connection to JIRA successful! Server version: {server_version}")
            return True, server_version
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except ValueError:
                    logger.error(f"Response content: {e.response.text}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Connection test failed with unexpected error: {str(e)}")
            return False, str(e)
    
    @cached(timeout=3600)  # Cache for 1 hour
    def get_issue(self, issue_key: str, fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific issue by its key.
        
        Args:
            issue_key: The issue key (e.g., DEMO-1)
            fields: Optional list of fields to include
            
        Returns:
            dict: Issue data or None if not found/error
        """
        try:
            # Build params
            params = {}
            if fields:
                params["fields"] = ",".join(fields)
            
            logger.info(f"Fetching issue: {issue_key}")
            response = requests.get(
                f"{self.base_url}/rest/api/2/issue/{issue_key}",
                auth=(self.username, self.api_token),
                headers=self.headers,
                params=params,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                issue = response.json()
                logger.info(f"Successfully retrieved issue: {issue_key}")
                return issue
            else:
                logger.error(f"Failed to get issue {issue_key}: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting issue {issue_key}: {str(e)}")
            return None
    
    @cached(timeout=3600)
    def search_issues(self, jql: str = None, max_results: int = 50, start_at: int = 0, 
                     fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for issues using JQL.
        
        Args:
            jql: JQL search string
            max_results: Maximum number of results to return
            start_at: Starting index for pagination
            fields: Optional list of fields to include
            
        Returns:
            list: List of issues
        """
        try:
            # Build params
            params = {
                "maxResults": max_results,
                "startAt": start_at
            }
            
            if jql:
                params["jql"] = jql
                
            if fields:
                params["fields"] = ",".join(fields)
            
            logger.info(f"Searching for issues with JQL: {jql}")
            response = requests.get(
                f"{self.base_url}/rest/api/2/search",
                auth=(self.username, self.api_token),
                headers=self.headers,
                params=params,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                search_results = response.json()
                issues = search_results.get('issues', [])
                logger.info(f"Search returned {len(issues)} issues")
                return issues
            else:
                logger.error(f"Failed to search issues: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error searching issues: {str(e)}")
            return []
    
    @cached(timeout=7200)  # Cache for 2 hours
    def get_all_issues(self, jql: str = None, fields: Optional[List[str]] = None, 
                      max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Get all issues matching a JQL query with pagination handling.
        
        Args:
            jql: JQL search string
            fields: Optional list of fields to include
            max_results: Maximum number of results per request
            
        Returns:
            list: List of all matching issues
        """
        all_issues = []
        start_at = 0
        
        logger.info(f"Retrieving all issues matching JQL: {jql}")
        
        while True:
            try:
                # Get batch of issues
                issues = self.search_issues(
                    jql=jql,
                    fields=fields,
                    max_results=max_results,
                    start_at=start_at
                )
                
                if not issues:
                    break
                    
                all_issues.extend(issues)
                
                # Check if we've reached the end
                if len(issues) < max_results:
                    break
                
                # Move to next page
                start_at += len(issues)
                logger.info(f"Retrieved {len(all_issues)} issues so far")
            except Exception as e:
                logger.error(f"Error retrieving all issues: {str(e)}")
                break
        
        logger.info(f"Retrieved a total of {len(all_issues)} issues")
        return all_issues
    
    @cached(timeout=3600)
    def get_issue_types(self) -> List[Dict[str, Any]]:
        """
        Get all issue types defined in the Jira instance.
        
        Returns:
            list: List of issue types
        """
        try:
            logger.info("Fetching issue types...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/issuetype",
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                issue_types = response.json()
                logger.info(f"Successfully retrieved {len(issue_types)} issue types")
                return issue_types
            else:
                logger.error(f"Failed to get issue types: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting issue types: {str(e)}")
            return []
    
    @cached(timeout=3600)
    def get_projects(self) -> List[Dict[str, Any]]:
        """
        Get all projects visible to the authenticated user.
        
        Returns:
            list: List of projects
        """
        try:
            logger.info("Fetching projects...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/project",
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                projects = response.json()
                logger.info(f"Successfully retrieved {len(projects)} projects")
                return projects
            else:
                logger.error(f"Failed to get projects: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting projects: {str(e)}")
            return []
    
    def create_issue(self, project_key: str, issue_type: str, summary: str, description: str, 
                    fields: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Create a new issue.
        
        Args:
            project_key: The project key
            issue_type: The issue type name or ID
            summary: The issue summary
            description: The issue description
            fields: Dictionary of additional fields to set
            
        Returns:
            dict: Created issue data or None if error
        """
        try:
            # Build issue data
            issue_data = {
                "fields": {
                    "project": {
                        "key": project_key
                    },
                    "issuetype": {
                        "name": issue_type
                    },
                    "summary": summary,
                    "description": description
                }
            }
            
            # Add additional fields if provided
            if fields:
                issue_data["fields"].update(fields)
            
            logger.info(f"Creating issue in project {project_key} of type {issue_type}")
            response = requests.post(
                f"{self.base_url}/rest/api/2/issue",
                auth=(self.username, self.api_token),
                headers=self.headers,
                json=issue_data,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()
            issue = response.json()
            logger.info(f"Successfully created issue: {issue.get('key')}")
            return issue
        except Exception as e:
            logger.error(f"Error creating issue: {str(e)}")
            return None
    
    def get_issue_content(self, issue_key: str) -> Optional[Dict[str, Any]]:
        """
        Get the content of an issue in a suitable format for RAG.
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            issue_key: The key of the issue
            
        Returns:
            dict: Processed issue content or None
        """
        try:
            # Get issue with all relevant fields
            fields = [
                "summary", "description", "issuetype", "status", "created", 
                "updated", "assignee", "reporter", "priority", "labels",
                "components", "fixVersions", "resolution", "comment"
            ]
            
            issue = self.get_issue(issue_key, fields=fields)
            
            if not issue:
                return None
            
            # Extract basic metadata
            fields = issue.get("fields", {})
            metadata = {
                "id": issue.get("id"),
                "key": issue.get("key"),
                "type": fields.get("issuetype", {}).get("name"),
                "url": f"{self.base_url}/browse/{issue.get('key')}",
                "created": fields.get("created"),
                "updated": fields.get("updated"),
                "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
                "reporter": fields.get("reporter", {}).get("displayName") if fields.get("reporter") else None,
                "priority": fields.get("priority", {}).get("name") if fields.get("priority") else None,
                "labels": fields.get("labels", []),
                "status": fields.get("status", {}).get("name") if fields.get("status") else None,
                "resolution": fields.get("resolution", {}).get("name") if fields.get("resolution") else None
            }
            
            # Extract people
            if fields.get("assignee"):
                metadata["assignee"] = fields.get("assignee", {}).get("displayName")
                
            if fields.get("reporter"):
                metadata["reporter"] = fields.get("reporter", {}).get("displayName")
            
            # Extract content fields
            content_parts = []
            
            # Add summary
            if fields.get("summary"):
                content_parts.append(f"Summary: {fields.get('summary')}")
            
            # Add description
            if fields.get("description"):
                # Handle potential Atlassian Document Format
                description = fields.get("description")
                if isinstance(description, dict):
                    # Try to extract text from ADF
                    try:
                        # Simple extraction of text content
                        plain_text = self._extract_text_from_adf(description)
                        content_parts.append(f"Description: {plain_text}")
                    except:
                        # Fallback to string representation
                        content_parts.append(f"Description: {str(description)}")
                else:
                    # Plain text or HTML
                    if "<" in description and ">" in description:
                        # Likely HTML
                        plain_text = extract_text_from_html(description)
                        content_parts.append(f"Description: {plain_text}")
                    else:
                        content_parts.append(f"Description: {description}")
            
            # Add comments
            comments = fields.get("comment", {}).get("comments", [])
            for comment in comments:
                author = comment.get("author", {}).get("displayName", "Unknown")
                created = comment.get("created", "")
                
                comment_body = comment.get("body", "")
                if isinstance(comment_body, dict):
                    # ADF format
                    comment_text = self._extract_text_from_adf(comment_body)
                elif "<" in comment_body and ">" in comment_body:
                    # HTML format
                    comment_text = extract_text_from_html(comment_body)
                else:
                    comment_text = comment_body
                
                content_parts.append(f"Comment by {author} on {created}: {comment_text}")
            
            # Combine all content
            full_content = "\n\n".join(content_parts)
            
            return {
                "metadata": metadata,
                "content": full_content
            }
        except Exception as e:
            logger.error(f"Error processing issue content: {str(e)}")
            return None
    
    def _extract_text_from_adf(self, adf_content: Dict[str, Any]) -> str:
        """
        Extract plain text from Atlassian Document Format (ADF).
        
        Args:
            adf_content: ADF content object
            
        Returns:
            Plain text representation
        """
        # Simple implementation - traverse content and extract text
        if not adf_content:
            return ""
        
        text_parts = []
        
        # Handle different ADF structures
        if "content" in adf_content:
            for node in adf_content.get("content", []):
                text_parts.append(self._extract_node_text(node))
        elif "version" in adf_content and "content" in adf_content:
            # Document node
            for node in adf_content.get("content", []):
                text_parts.append(self._extract_node_text(node))
        else:
            # Try to handle as a single node
            text_parts.append(self._extract_node_text(adf_content))
        
        return "\n".join(text_parts)
    
    def _extract_node_text(self, node: Dict[str, Any]) -> str:
        """
        Extract text from an ADF node.
        
        Args:
            node: ADF node
            
        Returns:
            Text from the node
        """
        node_type = node.get("type", "")
        
        if node_type == "text":
            return node.get("text", "")
        elif node_type == "paragraph":
            paragraph_parts = []
            for content_node in node.get("content", []):
                paragraph_parts.append(self._extract_node_text(content_node))
            return " ".join(paragraph_parts)
        elif node_type == "heading":
            heading_parts = []
            for content_node in node.get("content", []):
                heading_parts.append(self._extract_node_text(content_node))
            return f"\n{'#' * node.get('level', 1)} {' '.join(heading_parts)}\n"
        elif node_type == "bulletList" or node_type == "orderedList":
            list_parts = []
            for item in node.get("content", []):
                list_parts.append(self._extract_node_text(item))
            return "\n".join(list_parts)
        elif node_type == "listItem":
            item_parts = []
            for content_node in node.get("content", []):
                item_parts.append(self._extract_node_text(content_node))
            return f"- {' '.join(item_parts)}"
        elif node_type == "codeBlock":
            code_parts = []
            for content_node in node.get("content", []):
                code_parts.append(self._extract_node_text(content_node))
            return f"\n```\n{''.join(code_parts)}\n```\n"
        elif "content" in node:
            # Generic container node
            container_parts = []
            for content_node in node.get("content", []):
                container_parts.append(self._extract_node_text(content_node))
            return "\n".join(container_parts)
        else:
            # Unknown node type, return empty string
            return ""

# Global instance
jira_client = JIRAClient()















"""
JIRA connector for the RAG system.
"""
import logging
from typing import List, Dict, Any, Optional
from data_sources.jira.client import jira_client
from utils.cache import cached

logger = logging.getLogger(__name__)

class JIRAConnector:
    """
    Connector between JIRA and the RAG system.
    """
    
    def __init__(self):
        """Initialize the JIRA connector."""
        self.client = jira_client
        logger.info("Initialized JIRA connector")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to JIRA.
        
        Returns:
            Dict with connection status and details
        """
        success, details = self.client.test_connection()
        return {
            "success": success,
            "details": details,
            "source": "jira"
        }
    
    @cached(timeout=3600)  # Cache for 1 hour
    def get_projects(self) -> List[Dict[str, Any]]:
        """
        Get all JIRA projects.
        
        Returns:
            List of projects with key information
        """
        try:
            projects = self.client.get_projects()
            # Extract relevant information
            project_info = []
            for project in projects:
                project_info.append({
                    "id": project.get("id"),
                    "key": project.get("key"),
                    "name": project.get("name"),
                    "projectTypeKey": project.get("projectTypeKey"),
                    "simplified": project.get("simplified", False)
                })
            return project_info
        except Exception as e:
            logger.error(f"Error getting projects: {str(e)}")
            return []
    
    @cached(timeout=3600)  # Cache for 1 hour
    def search_issues(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for issues in JIRA.
        
        Args:
            query: Search query or JQL
            max_results: Maximum number of results
            
        Returns:
            List of matching issues
        """
        try:
            # Detect if the query is JQL or free text
            if " = " in query or " ~ " in query or " > " in query or " < " in query:
                # Looks like JQL
                jql = query
            else:
                # Free text search
                jql = f'text ~ "{query}"'
            
            # Get fields to retrieve
            fields = [
                "summary", "issuetype", "status", "created", 
                "updated", "assignee", "priority"
            ]
            
            # Search for issues
            issues = self.client.search_issues(
                jql=jql,
                max_results=max_results,
                fields=fields
            )
            
            # Format results
            formatted_issues = []
            for issue in issues:
                fields = issue.get("fields", {})
                formatted_issue = {
                    "id": issue.get("id"),
                    "key": issue.get("key"),
                    "summary": fields.get("summary"),
                    "type": fields.get("issuetype", {}).get("name") if fields.get("issuetype") else None,
                    "status": fields.get("status", {}).get("name") if fields.get("status") else None,
                    "created": fields.get("created"),
                    "updated": fields.get("updated"),
                    "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
                    "priority": fields.get("priority", {}).get("name") if fields.get("priority") else None,
                    "url": f"{self.client.base_url}/browse/{issue.get('key')}"
                }
                formatted_issues.append(formatted_issue)
            
            return formatted_issues
        except Exception as e:
            logger.error(f"Error searching issues: {str(e)}")
            return []
    
    def get_issue_for_rag(self, issue_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a single issue for RAG processing.
        
        Args:
            issue_key: Key of the issue
            
        Returns:
            Processed issue content or None
        """
        try:
            issue_content = self.client.get_issue_content(issue_key)
            if not issue_content:
                return None
            
            # Format for RAG
            return {
                "content": issue_content.get("content", ""),
                "metadata": issue_content.get("metadata", {})
            }
        except Exception as e:
            logger.error(f"Error getting issue for RAG: {str(e)}")
            return None
    
    def get_issues_for_rag(self, jql: str = None, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Get issues for RAG processing based on JQL.
        
        Args:
            jql: JQL query to filter issues
            max_results: Maximum number of issues to process
            
        Returns:
            List of processed issues
        """
        try:
            # Default JQL if none provided
            if not jql:
                jql = "updated >= -7d ORDER BY updated DESC"
            
            # Get issues matching the JQL
            issues = self.client.get_all_issues(jql=jql, max_results=max_results)
            
            # Process each issue
            rag_documents = []
            for issue in issues:
                issue_key = issue.get("key")
                issue_content = self.get_issue_for_rag(issue_key)
                if issue_content:
                    rag_documents.append(issue_content)
            
            logger.info(f"Retrieved {len(rag_documents)} JIRA issues for RAG")
            return rag_documents
        except Exception as e:
            logger.error(f"Error getting issues for RAG: {str(e)}")
            return []
    
    def get_project_issues_for_rag(self, project_key: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Get all issues in a project for RAG processing.
        
        Args:
            project_key: Key of the project
            max_results: Maximum number of issues to process
            
        Returns:
            List of processed issues
        """
        try:
            # JQL for project issues
            jql = f'project = "{project_key}" ORDER BY updated DESC'
            
            # Use the standard issue retrieval method
            return self.get_issues_for_rag(jql=jql, max_results=max_results)
        except Exception as e:
            logger.error(f"Error getting project issues for RAG: {str(e)}")
            return []

# Global instance
jira_connector = JIRAConnector()














"""
Remedy API client for the RAG system.
"""
import logging
import requests
import json
import urllib3
import getpass
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import quote
from utils.cache import cached
from config import Config

logger = logging.getLogger(__name__)

class RemedyClient:
    """
    Client for BMC Remedy REST API operations with comprehensive error handling.
    """
    
    def __init__(self, server_url: str = None, username: str = None, password: str = None, ssl_verify: bool = None):
        """
        Initialize the Remedy client with server and authentication details.
        
        Args:
            server_url: The base URL of the Remedy server
            username: Username for authentication
            password: Password for authentication
            ssl_verify: Whether to verify SSL certificates
        """
        self.server_url = server_url or Config.REMEDY_URL
        self.server_url = self.server_url.rstrip('/')
        self.username = username or Config.REMEDY_USERNAME
        self.password = password or Config.REMEDY_PASSWORD
        
        # Handle SSL verification
        if ssl_verify is None:
            ssl_verify = Config.REMEDY_SSL_VERIFY
        if not ssl_verify:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled.")
        self.ssl_verify = ssl_verify
        
        # Authentication tokens
        self.token = None
        self.token_type = "AR-JWT"
        
        logger.info(f"Initialized Remedy client for {self.server_url}")
    
    def login(self) -> Tuple[int, str]:
        """
        Log in to Remedy and get authentication token.
        
        Returns:
            tuple: (returnVal, token) where returnVal is 1 on success, -1 on failure
        """
        if not self.username:
            self.username = input("Enter Username: ")
        if not self.password:
            self.password = getpass.getpass(prompt="Enter Password: ")
        
        logger.info(f"Attempting to login as {self.username}")
        url = f"{self.server_url}/api/jwt/login"
        payload = {"username": self.username, "password": self.password}
        headers = {"content-type": "application/x-www-form-urlencoded"}
        
        try:
            r = requests.post(url, data=payload, headers=headers, verify=self.ssl_verify)
            if r.status_code == 200:
                self.token = r.text
                logger.info("Login successful")
                return 1, self.token
            else:
                logger.error(f"Login failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return -1, r.text
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return -1, str(e)
    
    def logout(self) -> bool:
        """
        Log out from Remedy and invalidate the token.
        
        Returns:
            bool: True on success, False on failure
        """
        if not self.token:
            logger.warning("Cannot logout: No active token")
            return False
        
        logger.info("Logging out and invalidating token")
        url = f"{self.server_url}/api/jwt/logout"
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        try:
            r = requests.post(url, headers=headers, verify=self.ssl_verify)
            if r.status_code == 204 or r.status_code == 200:
                logger.info("Logout successful")
                self.token = None
                return True
            else:
                logger.error(f"Logout failed with status code: {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False
    
    def _ensure_logged_in(self) -> bool:
        """
        Ensure the client is logged in, login if necessary.
        
        Returns:
            bool: True if logged in, False otherwise
        """
        if not self.token:
            status, _ = self.login()
            return status == 1
        return True
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the connection to Remedy API.
        
        Returns:
            tuple: (success, message)
        """
        try:
            logger.info("Testing connection to Remedy...")
            status, message = self.login()
            if status == 1:
                self.logout()
                return True, "Connection successful"
            else:
                return False, f"Login failed: {message}"
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False, str(e)
    
    @cached(timeout=3600)  # Cache for 1 hour
    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific incident by its ID.
        
        Args:
            incident_id: The Incident Number (e.g., INC000001482087)
            
        Returns:
            dict: Incident data or None if not found/error
        """
        if not self._ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return None
        
        logger.info(f"Fetching incident: {incident_id}")
        
        # Create qualified query
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incident data
        result = self.query_form("HPD:Help Desk", qualified_query, fields)
        
        if result and "entries" in result and len(result["entries"]) > 0:
            logger.info(f"Successfully retrieved incident: {incident_id}")
            return result["entries"][0]
        else:
            logger.error(f"Incident not found or error: {incident_id}")
            return None
    
    @cached(timeout=3600)
    def get_incidents_by_date(self, date: str, status: str = None, owner_group: str = None) -> List[Dict[str, Any]]:
        """
        Get all incidents submitted on a specific date.
        
        Args:
            date: The submission date in YYYY-MM-DD format
            status: Optional status filter (e.g., "Closed", "Open")
            owner_group: Optional owner group filter
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self._ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Fetching incidents for date: {date}")
        
        # Parse the date and create date range (entire day)
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            start_datetime = date_obj.strftime("%Y-%m-%d 00:00:00.000")
            end_datetime = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00.000")
            
            # Create qualified query
            query_parts = [f"'Submit Date' >= \"{start_datetime}\" AND 'Submit Date' < \"{end_datetime}\""]
            
            # Add status filter if provided
            if status:
                query_parts.append(f"'Status'=\"{status}\"")
            
            # Add owner group filter if provided
            if owner_group:
                query_parts.append(f"'Owner Group'=\"{owner_group}\"")
            
            qualified_query = " AND ".join(query_parts)
            
            # Fields to retrieve
            fields = [
                "Assignee", "Incident Number", "Description", "Status", "Owner",
                "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
                "Priority", "Environment", "Summary", "Support Group Name",
                "Request Assignee", "Work Order ID", "Request Manager"
            ]
            
            # Get the incidents
            result = self.query_form("HPD:Help Desk", qualified_query, fields)
            
            if result and "entries" in result:
                logger.info(f"Retrieved {len(result['entries'])} incidents for date {date}")
                return result["entries"]
            else:
                logger.warning(f"No incidents found for date {date} or error occurred")
                return []
        except ValueError:
            logger.error(f"Invalid date format: {date}. Use YYYY-MM-DD.")
            return []
    
    @cached(timeout=3600)
    def get_incidents_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get incidents by their status.
        
        Args:
            status: The status to filter by (e.g., "Open", "Closed", "Resolved")
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self._ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Fetching incidents with status: {status}")
        
        # Create qualified query
        qualified_query = f"'Status'=\"{status}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents with status {status}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found with status {status} or error occurred")
            return []
    
    @cached(timeout=3600)
    def get_incidents_by_assignee(self, assignee: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get incidents assigned to a specific person.
        
        Args:
            assignee: The assignee name
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self._ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Fetching incidents assigned to: {assignee}")
        
        # Create qualified query
        qualified_query = f"'Assignee'=\"{assignee}\""
        
        # Fields to retrieve
        fields = [
            "Assignee", "Incident Number", "Description", "Status", "Owner",
            "Submitter", "Impact", "Owner Group", "Submit Date", "Assigned Group",
            "Priority", "Environment", "Summary", "Support Group Name",
            "Request Assignee", "Work Order ID", "Request Manager"
        ]
        
        # Get the incidents
        result = self.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
        
        if result and "entries" in result:
            logger.info(f"Retrieved {len(result['entries'])} incidents assigned to {assignee}")
            return result["entries"]
        else:
            logger.warning(f"No incidents found assigned to {assignee} or error occurred")
            return []
    
    def query_form(self, form_name: str, qualified_query: str = None, fields: List[str] = None, limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Query a Remedy form with optional filters and field selection.
        
        Args:
            form_name: The name of the form to query (e.g., "HPD:Help Desk")
            qualified_query: Optional qualified query string for filtering
            fields: Optional list of fields to retrieve
            limit: Maximum number of records to retrieve
            
        Returns:
            dict: Query result or None if error
        """
        if not self._ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return None
        
        logger.info(f"Querying form: {form_name}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/{form_name}"
        
        # Build headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Build query parameters
        params = {}
        if qualified_query:
            params["q"] = qualified_query
        if fields:
            params["fields"] = ",".join(fields)
        if limit:
            params["limit"] = limit
        
        # Make the request
        try:
            r = requests.get(url, headers=headers, params=params, verify=self.ssl_verify)
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Successfully queried form {form_name} and got {len(result.get('entries', []))} results")
                return result
            else:
                logger.error(f"Query failed with status code: {r.status_code}")
                logger.error(f"Headers: {r.headers}")
                logger.error(f"Response: {r.text}")
                return None
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return None
    
    def create_incident(self, summary: str, description: str, impact: str = "4-Minor/Localized",
                       urgency: str = "4-Low", reported_source: str = "Direct Input", 
                       service_type: str = "User Service Restoration", assigned_group: str = None) -> Optional[Dict[str, Any]]:
        """
        Create a new incident in Remedy.
        
        Args:
            summary: Incident summary/title
            description: Detailed description
            impact: Impact level (1-5)
            urgency: Urgency level (1-4)
            reported_source: How the incident was reported
            service_type: Type of service
            assigned_group: Group to assign the incident to
            
        Returns:
            dict: Created incident data or None if error
        """
        if not self._ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return None
        
        logger.info(f"Creating new incident: {summary}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Help Desk"
        
        # Build headers
        headers = {
            "Authorization": f"{self.token_type} {self.token}",
            "Content-Type": "application/json"
        }
        
        # Build incident data
        incident_data = {
            "values": {
                "Summary": summary,
                "Description": description,
                "Impact": impact,
                "Urgency": urgency,
                "Reported Source": reported_source,
                "Service Type": service_type
            }
        }
        
        if assigned_group:
            incident_data["values"]["Assigned Group"] = assigned_group
        
        # Make the request
        try:
            r = requests.post(url, headers=headers, json=incident_data, verify=self.ssl_verify)
            if r.status_code == 201:
                result = r.json()
                logger.info(f"Successfully created incident: {result.get('values', {}).get('Incident Number')}")
                return result
            else:
                logger.error(f"Create incident failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return None
        except Exception as e:
            logger.error(f"Create incident error: {str(e)}")
            return None
    
    def update_incident(self, incident_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing incident.
        
        Args:
            incident_id: The Incident Number to update
            update_data: Dictionary of fields to update
            
        Returns:
            bool: True on success, False on failure
        """
        if not self._ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return False
        
        logger.info(f"Updating incident: {incident_id}")
        
        # Build URL
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Help Desk"
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        url = f"{url}?q={quote(qualified_query)}"
        
        # Build headers
        headers = {
            "Authorization": f"{self.token_type} {self.token}",
            "Content-Type": "application/json"
        }
        
        # Build update data
        payload = {
            "values": update_data
        }
        
        # Make the request
        try:
            r = requests.put(url, headers=headers, json=payload, verify=self.ssl_verify)
            if r.status_code == 204:
                logger.info(f"Successfully updated incident: {incident_id}")
                return True
            else:
                logger.error(f"Update incident failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return False
        except Exception as e:
            logger.error(f"Update incident error: {str(e)}")
            return False
    
    def get_incident_history(self, incident_id: str) -> List[Dict[str, Any]]:
        """
        Get the history of changes for a specific incident.
        
        Args:
            incident_id: The Incident Number
            
        Returns:
            list: History entries or empty list if none found/error
        """
        if not self._ensure_logged_in():
            logger.error("No authentication token. Please login first.")
            return []
        
        logger.info(f"Fetching history for incident: {incident_id}")
        
        # Build URL for history form
        url = f"{self.server_url}/api/arsys/v1/entry/HPD:Help Desk History"
        
        # Qualified query to filter by incident number
        qualified_query = f"'Incident Number'=\"{incident_id}\""
        
        # Headers
        headers = {"Authorization": f"{self.token_type} {self.token}"}
        
        # Query parameters
        params = {
            "q": qualified_query,
            "fields": "History Date Time,Action,Description,Status,Changed By,Assigned Group"
        }
        
        # Make the request
        try:
            r = requests.get(url, headers=headers, params=params, verify=self.ssl_verify)
            if r.status_code == 200:
                result = r.json()
                logger.info(f"Successfully retrieved history for incident {incident_id} with {len(result.get('entries', []))} entries")
                return result.get("entries", [])
            else:
                logger.error(f"Get history failed with status code: {r.status_code}")
                logger.error(f"Response: {r.text}")
                return []
        except Exception as e:
            logger.error(f"Get history error: {str(e)}")
            return []
    
    def process_incident_for_rag(self, incident: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process an incident into a forma
def process_incident_for_rag(self, incident: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process an incident into a format suitable for RAG indexing.
        
        Args:
            incident: Raw incident data from Remedy API
            
        Returns:
            dict: Processed incident with metadata and content
        """
        if not incident or "values" not in incident:
            return None
        
        values = incident.get("values", {})
        
        # Extract metadata
        metadata = {
            "incident_number": values.get("Incident Number"),
            "status": values.get("Status"),
            "priority": values.get("Priority"),
            "impact": values.get("Impact"),
            "assignee": values.get("Assignee"),
            "owner": values.get("Owner"),
            "owner_group": values.get("Owner Group"),
            "assigned_group": values.get("Assigned Group"),
            "submitter": values.get("Submitter"),
            "submit_date": values.get("Submit Date"),
            "summary": values.get("Summary")
        }
# Build content for embedding
        content_parts = []
        
        if values.get("Summary"):
            content_parts.append(f"Summary: {values.get('Summary')}")
        
        if values.get("Description"):
            content_parts.append(f"Description: {values.get('Description')}")
        
        if values.get("Status"):
            content_parts.append(f"Status: {values.get('Status')}")
        
        if values.get("Priority"):
            content_parts.append(f"Priority: {values.get('Priority')}")
        
        if values.get("Impact"):
            content_parts.append(f"Impact: {values.get('Impact')}")
        
        if values.get("Assignee"):
            content_parts.append(f"Assigned to: {values.get('Assignee')}")
        
        if values.get("Owner Group"):
            content_parts.append(f"Owner Group: {values.get('Owner Group')}")
        
        # Combine content parts into a single text
        content = "\n".join(content_parts)
        
        return {"metadata": metadata,
            "content": content,
            "raw_data": values
        }

# Global instance
remedy_client = RemedyClient()

















"""
Remedy connector for the RAG system.
"""
import logging
from typing import List, Dict, Any, Optional
from data_sources.remedy.client import remedy_client
from utils.cache import cached

logger = logging.getLogger(__name__)

class RemedyConnector:
    """
    Connector between Remedy and the RAG system.
    """
    
    def __init__(self):
        """Initialize the Remedy connector."""
        self.client = remedy_client
        logger.info("Initialized Remedy connector")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to Remedy.
        
        Returns:
            Dict with connection status and details
        """
        success, details = self.client.test_connection()
        return {
            "success": success,
            "details": details,
            "source": "remedy"
        }
    
    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific incident by ID.
        
        Args:
            incident_id: Incident number
            
        Returns:
            Dict with incident details or None
        """
        try:
            incident = self.client.get_incident(incident_id)
            if not incident:
                return None
            
            # Format incident for API response
            values = incident.get("values", {})
            formatted_incident = {
                "incident_number": values.get("Incident Number"),
                "summary": values.get("Summary"),
                "description": values.get("Description"),
                "status": values.get("Status"),
                "priority": values.get("Priority"),
                "impact": values.get("Impact"),
                "assignee": values.get("Assignee"),
                "owner": values.get("Owner"),
                "owner_group": values.get("Owner Group"),
                "assigned_group": values.get("Assigned Group"),
                "submitter": values.get("Submitter"),
                "submit_date": values.get("Submit Date"),
                "environment": values.get("Environment"),
                "support_group": values.get("Support Group Name"),
                "work_order_id": values.get("Work Order ID")
            }
            
            return formatted_incident
        except Exception as e:
            logger.error(f"Error getting incident: {str(e)}")
            return None
    
    @cached(timeout=3600)  # Cache for 1 hour
    def search_incidents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for incidents in Remedy.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching incidents
        """
        try:
            # Create query based on search terms
            qualified_query = f"'Summary' LIKE \"%{query}%\" OR 'Description' LIKE \"%{query}%\""
            
            # Fields to retrieve
            fields = [
                "Incident Number", "Summary", "Status", "Priority", 
                "Assignee", "Submit Date", "Owner Group"
            ]
            
            # Query the form
            result = self.client.query_form("HPD:Help Desk", qualified_query, fields, limit=limit)
            
            if not result or "entries" not in result:
                return []
            
            # Format results
            incidents = []
            for entry in result.get("entries", []):
                values = entry.get("values", {})
                incident = {
                    "incident_number": values.get("Incident Number"),
                    "summary": values.get("Summary"),
                    "status": values.get("Status"),
                    "priority": values.get("Priority"),
                    "assignee": values.get("Assignee"),
                    "submit_date": values.get("Submit Date"),
                    "owner_group": values.get("Owner Group")
                }
                incidents.append(incident)
            
            return incidents
        except Exception as e:
            logger.error(f"Error searching incidents: {str(e)}")
            return []
    
    def get_incidents_by_status(self, status: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get incidents by status.
        
        Args:
            status: Status to filter by
            limit: Maximum number of results
            
        Returns:
            List of incidents
        """
        try:
            incidents = self.client.get_incidents_by_status(status, limit=limit)
            
            # Format results
            formatted_incidents = []
            for incident in incidents:
                values = incident.get("values", {})
                formatted_incident = {
                    "incident_number": values.get("Incident Number"),
                    "summary": values.get("Summary"),
                    "status": values.get("Status"),
                    "priority": values.get("Priority"),
                    "assignee": values.get("Assignee"),
                    "submit_date": values.get("Submit Date"),
                    "owner_group": values.get("Owner Group")
                }
                formatted_incidents.append(formatted_incident)
            
            return formatted_incidents
        except Exception as e:
            logger.error(f"Error getting incidents by status: {str(e)}")
            return []
    
    def get_incident_for_rag(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single incident for RAG processing.
        
        Args:
            incident_id: Incident number
            
        Returns:
            Processed incident content or None
        """
        try:
            incident = self.client.get_incident(incident_id)
            if not incident:
                return None
            
            # Process for RAG
            processed_incident = self.client.process_incident_for_rag(incident)
            
            # Format for RAG
            return {
                "content": processed_incident.get("content", ""),
                "metadata": processed_incident.get("metadata", {})
            }
        except Exception as e:
            logger.error(f"Error getting incident for RAG: {str(e)}")
            return None
    
    def get_incidents_for_rag(self, status: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get incidents for RAG processing.
        
        Args:
            status: Optional status filter
            limit: Maximum number of incidents to process
            
        Returns:
            List of processed incidents
        """
        try:
            # Get raw incidents
            if status:
                incidents = self.client.get_incidents_by_status(status, limit=limit)
            else:
                # Get recent incidents (created within last 30 days)
                date_range = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                incidents = self.client.get_incidents_by_date(date_range, limit=limit)
            
            # Process each incident
            rag_documents = []
            for incident in incidents:
                processed_incident = self.client.process_incident_for_rag(incident)
                if processed_incident:
                    rag_document = {
                        "content": processed_incident.get("content", ""),
                        "metadata": processed_incident.get("metadata", {})
                    }
                    rag_documents.append(rag_document)
            
            logger.info(f"Retrieved {len(rag_documents)} Remedy incidents for RAG")
            return rag_documents
        except Exception as e:
            logger.error(f"Error getting incidents for RAG: {str(e)}")
            return []
    
    def create_incident(self, summary: str, description: str, impact: str = "4-Minor/Localized",
                      urgency: str = "4-Low") -> Optional[Dict[str, Any]]:
        """
        Create a new incident.
        
        Args:
            summary: Incident summary
            description: Incident description
            impact: Impact level
            urgency: Urgency level
            
        Returns:
            Dict with created incident details or None
        """
        try:
            # Create the incident
            result = self.client.create_incident(
                summary=summary,
                description=description,
                impact=impact,
                urgency=urgency
            )
            
            if not result:
                return None
            
            # Format the response
            values = result.get("values", {})
            created_incident = {
                "incident_number": values.get("Incident Number"),
                "summary": summary,
                "description": description,
                "impact": impact,
                "urgency": urgency,
                "status": values.get("Status")
            }
            
            return created_incident
        except Exception as e:
            logger.error(f"Error creating incident: {str(e)}")
            return None

# Global instance
from datetime import datetime, timedelta
remedy_connector = RemedyConnector()















"""
API endpoints for the RAG system.
"""













"""
Response formatting utilities for API endpoints.
"""
from typing import Dict, List, Any, Optional
from flask import jsonify

def format_error(message: str, status_code: int = 400) -> Dict[str, Any]:
    """
    Format an error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        
    Returns:
        Formatted error response
    """
    return {
        "success": False,
        "error": {
            "message": message,
            "status_code": status_code
        }
    }

def format_success(data: Any = None, message: str = None) -> Dict[str, Any]:
    """
    Format a success response.
    
    Args:
        data: Response data
        message: Optional success message
        
    Returns:
        Formatted success response
    """
    response = {
        "success": True
    }
    
    if data is not None:
        response["data"] = data
    
    if message:
        response["message"] = message
    
    return response

def format_rag_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a RAG response for API output.
    
    Args:
        response: Raw RAG response
        
    Returns:
        Formatted RAG response
    """
    return {
        "success": response.get("success", True),
        "query": response.get("query", ""),
        "response": response.get("response", ""),
        "sources": response.get("sources", [])
    }

def format_connection_status(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format connection status results for multiple data sources.
    
    Args:
        results: List of connection test results
        
    Returns:
        Formatted connection status
    """
    all_success = all(result.get("success", False) for result in results)
    
    status_by_source = {}
    for result in results:
        source = result.get("source", "unknown")
        status_by_source[source] = {
            "success": result.get("success", False),
            "details": result.get("details", "")
        }
    
    return {
        "success": all_success,
        "sources": status_by_source
    }

















"""
API routes for the RAG system.
"""
import logging
import json
from typing import Dict, List, Any
from flask import Blueprint, request, jsonify, Response, stream_with_context

from rag_engine.processor import rag_processor
from data_sources.confluence.connector import confluence_connector
from data_sources.jira.connector import jira_connector
from data_sources.remedy.connector import remedy_connector
from api.response_formatter import format_error, format_success, format_rag_response, format_connection_status

logger = logging.getLogger(__name__)

# Create Blueprint
api = Blueprint('api', __name__, url_prefix='/api')

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify(format_success(message="Service is running"))

@api.route('/status', methods=['GET'])
def connection_status():
    """Check status of all data source connections."""
    try:
        # Test connections to all data sources
        results = [
            confluence_connector.test_connection(),
            jira_connector.test_connection(),
            remedy_connector.test_connection()
        ]
        
        # Format results
        status = format_connection_status(results)
        
        return jsonify(format_success(data=status))
    except Exception as e:
        logger.error(f"Error checking connections: {str(e)}")
        return jsonify(format_error(f"Error checking connections: {str(e)}")), 500

@api.route('/query', methods=['POST'])
def query():
    """Process a query using the RAG system."""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify(format_error("Missing 'query' in request")), 400
        
        query_text = data.get('query')
        sources = data.get('sources', None)
        top_k = data.get('top_k', None)
        
        # Process the query
        result = rag_processor.process_query(query_text, sources=sources, top_k=top_k)
        
        # Format the response
        response = format_rag_response(result)
        
        return jsonify(format_success(data=response))
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify(format_error(f"Error processing query: {str(e)}")), 500

@api.route('/query/stream', methods=['POST'])
def stream_query():
    """Process a query and stream the response."""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify(format_error("Missing 'query' in request")), 400
        
        query_text = data.get('query')
        sources = data.get('sources', None)
        top_k = data.get('top_k', None)
        
        # Process the query with streaming
        response_stream, sources_list = rag_processor.process_query_stream(
            query_text, sources=sources, top_k=top_k
        )
        
        # Create a streaming response
        def generate():
            # First, send the sources as a separate JSON message
            sources_json = json.dumps({"type": "sources", "sources": sources_list})
            yield f"data: {sources_json}\n\n"
            
            # Then stream the response chunks
            for chunk in response_stream:
                if chunk:
                    chunk_json = json.dumps({"type": "chunk", "content": chunk})
                    yield f"data: {chunk_json}\n\n"
            
            # Signal completion
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
    except Exception as e:
        logger.error(f"Error streaming query: {str(e)}")
        error_json = json.dumps({"type": "error", "message": str(e)})
        return Response(
            f"data: {error_json}\n\n",
            mimetype='text/event-stream',
            status=500
        )

@api.route('/index', methods=['POST'])
def index_documents():
    """Index documents from data sources."""
    try:
        data = request.get_json()
        
        sources = data.get('sources', ['confluence', 'jira', 'remedy'])
        force_rebuild = data.get('force_rebuild', False)
        
        documents = []
        
        # Collect documents from each source
        if 'confluence' in sources:
            confluence_docs = confluence_connector.get_all_pages_for_rag(limit=100)
            documents.extend(confluence_docs)
            logger.info(f"Collected {len(confluence_docs)} documents from Confluence")
        
        if 'jira' in sources:
            jira_docs = jira_connector.get_issues_for_rag(max_results=100)
            documents.extend(jira_docs)
            logger.info(f"Collected {len(jira_docs)} documents from JIRA")
        
        if 'remedy' in sources:
            remedy_docs = remedy_connector.get_incidents_for_rag(limit=100)
            documents.extend(remedy_docs)
            logger.info(f"Collected {len(remedy_docs)} documents from Remedy")
        
        # Process and index documents
        success = rag_processor.process_documents(documents, force_rebuild=force_rebuild)
        
        if success:
            return jsonify(format_success(
                data={"indexed_count": len(documents)},
                message=f"Successfully indexed {len(documents)} documents"
            ))
        else:
            return jsonify(format_error("Failed to index documents")), 500
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        return jsonify(format_error(f"Error indexing documents: {str(e)}")), 500

# Confluence endpoints
@api.route('/confluence/spaces', methods=['GET'])
def get_confluence_spaces():
    """Get all Confluence spaces."""
    try:
        spaces = confluence_connector.get_spaces()
        return jsonify(format_success(data=spaces))
    except Exception as e:
        logger.error(f"Error getting Confluence spaces: {str(e)}")
        return jsonify(format_error(f"Error getting Confluence spaces: {str(e)}")), 500

@api.route('/confluence/search', methods=['GET'])
def search_confluence():
    """Search Confluence pages."""
    try:
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify(format_error("Missing 'query' parameter")), 400
        
        pages = confluence_connector.search_pages(query, limit=limit)
        return jsonify(format_success(data=pages))
    except Exception as e:
        logger.error(f"Error searching Confluence: {str(e)}")
        return jsonify(format_error(f"Error searching Confluence: {str(e)}")), 500

# JIRA endpoints
@api.route('/jira/projects', methods=['GET'])
def get_jira_projects():
    """Get all JIRA projects."""
    try:
        projects = jira_connector.get_projects()
        return jsonify(format_success(data=projects))
    except Exception as e:
        logger.error(f"Error getting JIRA projects: {str(e)}")
        return jsonify(format_error(f"Error getting JIRA projects: {str(e)}")), 500

@api.route('/jira/search', methods=['GET'])
def search_jira():
    """Search JIRA issues."""
    try:
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify(format_error("Missing 'query' parameter")), 400
        
        issues = jira_connector.search_issues(query, max_results=limit)
        return jsonify(format_success(data=issues))
    except Exception as e:
        logger.error(f"Error searching JIRA: {str(e)}")
        return jsonify(format_error(f"Error searching JIRA: {str(e)}")), 500

# Remedy endpoints
@api.route('/remedy/incidents', methods=['GET'])
def get_remedy_incidents():
    """Get Remedy incidents by status."""
    try:
        status = request.args.get('status', 'Open')
        limit = int(request.args.get('limit', 10))
        
        incidents = remedy_connector.get_incidents_by_status(status, limit=limit)
        return jsonify(format_success(data=incidents))
    except Exception as e:
        logger.error(f"Error getting Remedy incidents: {str(e)}")
        return jsonify(format_error(f"Error getting Remedy incidents: {str(e)}")), 500

@api.route('/remedy/search', methods=['GET'])
def search_remedy():
    """Search Remedy incidents."""
    try:
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify(format_error("Missing 'query' parameter")), 400
        
        incidents = remedy_connector.search_incidents(query, limit=limit)
        return jsonify(format_success(data=incidents))
    except Exception as e:
        logger.error(f"Error searching Remedy: {str(e)}")
        return jsonify(format_error(f"Error searching Remedy: {str(e)}")), 500





























"""
Main Flask application entry point.
"""
import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for

from config import Config
from utils.logger import setup_logger
from utils.cache import init_cache

from api.routes import api
from rag_engine.retrieval import retrieval_engine

logger = setup_logger()

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(Config)
    app.secret_key = Config.SECRET_KEY
    
    # Initialize cache
    init_cache(app)
    
    # Register blueprints
    app.register_blueprint(api)
    
    # Configure routes
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/chat')
    def chat():
        return render_template('chat.html')
    
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error.html', error=str(e), status_code=404), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return render_template('error.html', error=str(e), status_code=500), 500
    
    # Initialize RAG system
    try:
        # Check if index exists
        if not os.path.exists('indexes/document_index.faiss'):
            logger.info("No existing index found. You may need to run indexing first.")
        else:
            # Initialize retrieval engine with existing index
            retrieval_engine.initialize([])
            logger.info("Retrieval engine initialized with existing index")
    except Exception as e:
        logger.error(f"Error initializing retrieval engine: {str(e)}")
    
    return app

# Create application instance
app = create_app()

















a"""
Application launcher.
"""
import logging
from app import app
from config import Config

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info(f"Starting server on {Config.HOST}:{Config.PORT}")
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )



















a<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Enterprise RAG{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
    {% block additional_head %}{% endblock %}
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container-fluid">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="fas fa-robot"></i> Enterprise RAG
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item">
                        <a class="nav-link {% if request.path == url_for('index') %}active{% endif %}" 
                           href="{{ url_for('index') }}">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link {% if request.path == url_for('chat') %}active{% endif %}" 
                           href="{{ url_for('chat') }}">Chat</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        {% block content %}{% endblock %}
    </div>

    <footer class="footer mt-5 py-3 bg-light">
        <div class="container text-center">
            <span class="text-muted">Enterprise RAG System &copy; {% now 'Y' %}</span>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked@4.0.0/marked.min.js"></script>
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block additional_scripts %}{% endblock %}
</body>
</html>




















{% extends "base.html" %}

{% block title %}Enterprise RAG - Home{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8 offset-md-2 text-center">
        <div class="card shadow mb-5">
            <div class="card-body p-5">
                <h1 class="mb-4">Welcome to Enterprise RAG</h1>
                <p class="lead">
                    Your intelligent assistant for accessing information from Confluence, JIRA, and Remedy.
                </p>
                <hr class="my-4">
                <p>
                    This system uses Retrieval-Augmented Generation (RAG) to provide accurate answers
                    to your questions based on your enterprise data sources.
                </p>
                <div class="mt-5">
                    <a href="{{ url_for('chat') }}" class="btn btn-primary btn-lg">
                        <i class="fas fa-comments"></i> Start Chatting
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="row mt-4">
    <div class="col-md-4">
        <div class="card shadow h-100">
            <div class="card-body text-center">
                <i class="fab fa-confluence fa-4x text-primary mb-3"></i>
                <h3>Confluence Integration</h3>
                <p>
                    Access information from your Confluence pages, spaces, and documentation.
                </p>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card shadow h-100">
            <div class="card-body text-center">
                <i class="fab fa-jira fa-4x text-primary mb-3"></i>
                <h3>JIRA Integration</h3>
                <p>
                    Query information about issues, projects, and tasks tracked in JIRA.
                </p>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card shadow h-100">
            <div class="card-body text-center">
                <i class="fas fa-ticket-alt fa-4x text-primary mb-3"></i>
                <h3>Remedy Integration</h3>
                <p>
                    Get details about incidents, tickets, and their resolutions from Remedy.
                </p>
            </div>
        </div>
    </div>
</div>

<div class="row mt-5">
    <div class="col-md-8 offset-md-2">
        <div class="card shadow">
            <div class="card-header bg-secondary text-white">
                <h4 class="mb-0">System Status</h4>
            </div>
            <div class="card-body">
                <div id="status-container">
                    <div class="d-flex justify-content-center my-5">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block additional_scripts %}
<script>
    document.addEventListener('DOMContentLoaded', function() {
        // Check system status
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                const statusContainer = document.getElementById('status-container');
                
                if (data.success && data.data) {
                    const sources = data.data.sources;
                    let html = '<table class="table table-striped">';
                    html += '<thead><tr><th>Data Source</th><th>Status</th><th>Details</th></tr></thead>';
                    html += '<tbody>';
                    
                    for (const [source, status] of Object.entries(sources)) {
                        const statusClass = status.success ? 'text-success' : 'text-danger';
                        const statusIcon = status.success ? 
                            '<i class="fas fa-check-circle"></i>' : 
                            '<i class="fas fa-times-circle"></i>';
                        
                        html += `<tr>
                            <td>${source}</td>
                            <td class="${statusClass}">${statusIcon} ${status.success ? 'Connected' : 'Error'}</td>
                            <td>${status.details}</td>
                        </tr>`;
                    }
                    
                    html += '</tbody></table>';
                    statusContainer.innerHTML = html;
                } else {
                    statusContainer.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-triangle"></i> 
                            Failed to retrieve system status.
                        </div>
                    `;
                }
            })
            .catch(error => {
                const statusContainer = document.getElementById('status-container');
                statusContainer.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle"></i> 
                        Error checking system status: ${error.message}
                    </div>
                `;
            });
    });
</script>
{% endblock %}














{% extends "base.html" %}

{% block title %}Enterprise RAG - Chat{% endblock %}

{% block additional_head %}
<link rel="stylesheet" href="{{ url_for('static', filename='css/chat.css') }}">
{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-3">
        <div class="card shadow h-100">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">Data Sources</h5>
            </div>
            <div class="card-body">
                <div class="form-check form-switch mb-2">
                    <input class="form-check-input" type="checkbox" id="source-confluence" checked>
                    <label class="form-check-label" for="source-confluence">
                        <i class="fab fa-confluence"></i> Confluence
                    </label>
                </div>
                <div class="form-check form-switch mb-2">
                    <input class="form-check-input" type="checkbox" id="source-jira" checked>
                    <label class="form-check-label" for="source-jira">
                        <i class="fab fa-jira"></i> JIRA
                    </label>
                </div>
                <div class="form-check form-switch mb-4">
                    <input class="form-check-input" type="checkbox" id="source-remedy" checked>
                    <label class="form-check-label" for="source-remedy">
                        <i class="fas fa-ticket-alt"></i> Remedy
                    </label>
                </div>
                
                <hr>
                
                <div class="mb-3">
                    <label for="num-results" class="form-label">Number of Results: <span id="num-results-value">5</span></label>
                    <input type="range" class="form-range" id="num-results" min="1" max="10" value="5">
                </div>
                
                <button id="btn-index" class="btn btn-sm btn-outline-primary mt-3">
                    <i class="fas fa-sync"></i> Refresh Index
                </button>
            </div>
        </div>
    </div>
    
    <div class="col-md-9">
        <div class="card shadow chat-container">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">Chat with Enterprise RAG</h5>
            </div>
            <div class="card-body p-0">
                <div id="chat-messages" class="chat-messages">
                    <div class="message system">
                        <div class="message-content">
                            <p>👋 Hello! I'm your enterprise assistant. I can help you find information from Confluence, JIRA, and Remedy. What would you like to know?</p>
                        </div>
                    </div>
                </div>
                
                <div class="chat-input-container p-3 border-top">
                    <form id="chat-form">
                        <div class="input-group">
                            <input type="text" id="user-input" class="form-control" placeholder="Type your question here..." autocomplete="off">
                            <button type="submit" class="btn btn-primary">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="modal fade" id="sources-modal" tabindex="-1">
    <div class="modal-dialog modal-lg">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Information Sources</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div class="modal-body" id="sources-content">
                <div class="d-flex justify-content-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
            </div>
        </div>
    </div>
</div>

<div class="position-fixed bottom-0 end-0 p-3" style="z-index: 5">
    <div id="index-toast" class="toast hide" role="alert" aria-live="assertive" aria-atomic="true">
        <div class="toast-header">
            <strong class="me-auto">System Message</strong>
            <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body" id="index-toast-body">
        </div>
    </div>
</div>
{% endblock %}

{% block additional_scripts %}
<script src="{{ url_for('static', filename='js/chat.js') }}"></script>
{% endblock %}




















{% extends "base.html" %}

{% block title %}Error - Enterprise RAG{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8 offset-md-2 text-center">
        <div class="card shadow">
            <div class="card-body p-5">
                <h1 class="text-danger">
                    <i class="fas fa-exclamation-triangle"></i> Error {{ status_code }}
                </h1>
                <p class="lead mt-4">
                    {{ error }}
                </p>
                <hr class="my-4">
                <a href="{{ url_for('index') }}" class="btn btn-primary">
                    <i class="fas fa-home"></i> Return to Home
                </a>
            </div>
        </div>
    </div>
</div>
{% endblock %}




















/* Main CSS for Enterprise RAG */

:root {
    --primary-color: #4a6baf;
    --secondary-color: #5a7ac0;
    --accent-color: #7890d1;
    --light-color: #f8f9fa;
    --dark-color: #343a40;
    --success-color: #28a745;
    --info-color: #17a2b8;
    --warning-color: #ffc107;
    --danger-color: #dc3545;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f7fa;
    color: #333;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

.bg-primary {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
}

.bg-secondary {
    background: linear-gradient(135deg, var(--secondary-color), var(--accent-color)) !important;
}

.navbar-brand {
    font-weight: bold;
    letter-spacing: 0.5px;
}

.container-fluid {
    max-width: 1400px;
}

.card {
    border-radius: 0.5rem;
    border: none;
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
}

.card-header {
    border-top-left-radius: 0.5rem !important;
    border-top-right-radius: 0.5rem !important;
}

.btn-primary {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    border: none;
}

.btn-primary:hover {
    background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
}

.btn-outline-primary {
    color: var(--primary-color);
    border-color: var(--primary-color);
}

.btn-outline-primary:hover {
    background-color: var(--primary-color);
    color: white;
}

.shadow {
    box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15) !important;
}

code {
    background-color: #f0f0f0;
    padding: 0.2rem 0.4rem;
    border-radius: 0.25rem;
}

pre {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.25rem;
    border: 1px solid #e9ecef;
}

.footer {
    margin-top: auto;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.fade-in {
    animation: fadeIn 0.5s ease-in-out;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .container-fluid {
        padding: 0.5rem;
    }
    
    .card {
        margin-bottom: 1rem;
    }
}














/* Chat interface styling */

.chat-container {
    height: calc(100vh - 150px);
    min-height: 500px;
    display: flex;
    flex-direction: column;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
}

.message {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    max-width: 85%;
    animation: fadeIn 0.3s ease-in-out;
}

.message.user {
    align-self: flex-end;
    margin-left: auto;
}

.message.assistant {
    align-self: flex-start;
    margin-right: auto;
}

.message.system {
    align-self: center;
    max-width: 70%;
    margin: 1rem auto;
}

.message-content {
    padding: 0.75rem 1rem;
    border-radius: 1rem;
    position: relative;
}

.message.user .message-content {
    background: linear-gradient(135deg, #4a6baf, #5a7ac0);
    color: white;
    border-top-right-radius: 0;
}

.message.assistant .message-content {
    background-color: #f0f2f5;
    border-top-left-radius: 0;
}

.message.system .message-content {
    background-color: #fff3cd;
    border: 1px solid #ffeeba;
}

.message-content p {
    margin-bottom: 0.5rem;
}

.message-content p:last-child {
    margin-bottom: 0;
}

.message-meta {
    font-size: 0.75rem;
    color: #6c757d;
    margin-top: 0.25rem;
    display: flex;
    align-items: center;
}

.message.user .message-meta {
    justify-content: flex-end;
}

.chat-input-container {
    background-color: #fff;
}

#user-input {
    border-radius: 1.5rem;
    padding: 0.75rem 1rem;
    border: 1px solid #ced4da;
}

#user-input:focus {
    box-shadow: none;
    border-color: #4a6baf;
}

.btn-send {
    border-radius: 50%;
    width: 2.5rem;
    height: 2.5rem;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}

.typing-indicator {
    display: flex;
    align-items: center;
    padding: 0.5rem 1rem;
    background-color: #f0f2f5;
    border-radius: 1rem;
    margin-bottom: 1rem;
    width: fit-content;
}

.typing-indicator span {
    height: 0.5rem;
    width: 0.5rem;
    background-color: #6c757d;
    border-radius: 50%;
    display: inline-block;
    margin: 0 0.1rem;
    opacity: 0.6;
}

.typing-indicator span:nth-child(1) {
    animation: typingAnimation 1s infinite 0s;
}

.typing-indicator span:nth-child(2) {
    animation: typingAnimation 1s infinite 0.2s;
}

.typing-indicator span:nth-child(3) {
    animation: typingAnimation 1s infinite 0.4s;
}

@keyframes typingAnimation {
    0% {
        transform: translateY(0);
    }
    50% {
        transform: translateY(-0.5rem);
    }
    100% {
        transform: translateY(0);
    }
}

.source-item {
    margin-bottom: 0.75rem;
    padding: 0.75rem;
    background-color: #f8f9fa;
    border-radius: 0.5rem;
    border-left: 3px solid #4a6baf;
}

.source-title {
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.source-meta {
    font-size: 0.8rem;
    color: #6c757d;
}

.source-score {
    font-size: 0.8rem;
    padding: 0.1rem 0.5rem;
    border-radius: 1rem;
    background-color: #e9ecef;
}

.markdown-content code {
    background-color: #f0f0f0;
    padding: 0.2rem 0.4rem;
    border-radius: 0.25rem;
}

.markdown-content pre {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.25rem;
    border: 1px solid #e9ecef;
    overflow-x: auto;
}

.markdown-content table {
    border-collapse: collapse;
    margin: 1rem 0;
    width: 100%;
}

.markdown-content th, 
.markdown-content td {
    border: 1px solid #dee2e6;
    padding: 0.5rem;
}

.markdown-content th {
    background-color: #f0f2f5;
}

.markdown-content ul, 
.markdown-content ol {
    padding-left: 2rem;
}

.markdown-content img {
    max-width: 100%;
    height: auto;
}

.markdown-content blockquote {
    border-left: 3px solid #dee2e6;
    padding-left: 1rem;
    color: #6c757d;
}

/* Sources button */
.sources-btn {
    font-size: 0.8rem;
    padding: 0.2rem 0.5rem;
    margin-top: 0.5rem;
    background-color: #f0f2f5;
    border: 1px solid #dee2e6;
    border-radius: 0.3rem;
    color: #6c757d;
    cursor: pointer;
}

.sources-btn:hover {
    background-color: #e9ecef;
}

@media (max-width: 768px) {
    .chat-container {
        height: calc(100vh - 200px);
    }
    
    .message {
        max-width: 95%;
    }
    
    .message.system {
        max-width: 90%;
    }
}

















/**
 * Main JavaScript file for Enterprise RAG
 */

// Utility function to format dates in a human-readable format
function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleString();
}

// Utility function to safely convert Markdown to HTML
function markdownToHtml(markdown) {
    if (!markdown) return '';
    try {
        // Use marked library for conversion
        return marked.parse(markdown, {
            breaks: true,
            gfm: true,
            sanitize: true
        });
    } catch (error) {
        console.error('Error converting markdown to HTML:', error);
        return markdown;
    }
}

// Utility function to create a toast notification
function showToast(message, type = 'info') {
    // Create toast element if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    const toastId = 'toast-' + Date.now();
    const toastClass = `toast text-white bg-${type}`;
    
    const toastHtml = `
        <div id="${toastId}" class="${toastClass}" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header">
                <strong class="me-auto">
                    ${type === 'success' ? 'Success' : type === 'danger' ? 'Error' : 'Notification'}
                </strong>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {
        autohide: true,
        delay: 5000
    });
    
    toast.show();
    
    // Remove toast element after it's hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

// Utility function for making API requests
async function apiRequest(endpoint, method = 'GET', body = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        if (body) {
            options.body = JSON.stringify(body);
        }
        
        const response = await fetch(endpoint, options);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error?.message || 'An error occurred');
        }
        
        return data;
    } catch (error) {
        console.error(`API request error (${endpoint}):`, error);
        throw error;
    }
}

// Initialize popovers and tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Initialize all tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});



















/**
 * Chat functionality for Enterprise RAG
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const sourceConfluence = document.getElementById('source-confluence');
    const sourceJira = document.getElementById('source-jira');
    const sourceRemedy = document.getElementById('source-remedy');
    const numResults = document.getElementById('num-results');
    const numResultsValue = document.getElementById('num-results-value');
    const btnIndex = document.getElementById('btn-index');
    const sourcesModal = new bootstrap.Modal(document.getElementById('sources-modal'));
    const indexToast = new bootstrap.Toast(document.getElementById('index-toast'));
    const indexToastBody = document.getElementById('index-toast-body');
    
    // Chat state
    let chatHistory = [];
    let lastSources = [];
    let isAwaitingResponse = false;
    
    // Initialize chat
    initChat();
    
    // Event listeners
    chatForm.addEventListener('submit', handleChatSubmit);
    numResults.addEventListener('input', handleNumResultsChange);
    btnIndex.addEventListener('click', handleIndexRefresh);
    
    // Update num results value
    function handleNumResultsChange() {
        numResultsValue.textContent = numResults.value;
    }
    
    // Initialize chat interface
    function initChat() {
        // Scroll to bottom of chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Enable form
        userInput.disabled = false;
        userInput.focus();
    }
    
    // Handle chat form submission
    async function handleChatSubmit(event) {
        event.preventDefault();
        
        const query = userInput.value.trim();
        if (!query || isAwaitingResponse) return;
        
        // Add user message to chat
        addMessage(query, 'user');
        
        // Clear input
        userInput.value = '';
        
        // Get selected sources
        const sources = [];
        if (sourceConfluence.checked) sources.push('confluence');
        if (sourceJira.checked) sources.push('jira');
        if (sourceRemedy.checked) sources.push('remedy');
        
        // Get number of results
        const top_k = parseInt(numResults.value);
        
        // Disable input while waiting for response
        isAwaitingResponse = true;
        userInput.disabled = true;
        
        // Add typing indicator
        const typingIndicatorId = addTypingIndicator();
        
        try {
            // Use streaming endpoint
            const response = await fetch('/api/query/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query,
                    sources,
                    top_k
                })
            });
            
            // Process streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let responseText = '';
            let responseElement = null;
            
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                // Decode and buffer the chunk
                buffer += decoder.decode(value, { stream: true });
                
                // Process complete SSE messages
                const messages = buffer.split('\n\n');
                buffer = messages.pop() || ''; // Keep the last incomplete message in the buffer
                
                for (const message of messages) {
                    if (!message.trim() || !message.startsWith('data: ')) continue;
                    
                    try {
                        // Parse the SSE data
                        const data = JSON.parse(message.substring(6));
                        
                        if (data.type === 'sources') {
                            // Store sources for later display
                            lastSources = data.sources;
                        } else if (data.type === 'chunk') {
                            // First chunk? Create the response element
                            if (!responseElement) {
                                // Remove typing indicator
                                removeTypingIndicator(typingIndicatorId);
                                
                                // Create assistant message
                                responseElement = createMessageElement('', 'assistant');
                                chatMessages.appendChild(responseElement);
                                
                                // Create content element
                                const contentElement = document.createElement('div');
                                contentElement.className = 'message-content markdown-content';
                                responseElement.appendChild(contentElement);
                            }
                            
                            // Append chunk to response text
                            responseText += data.content;
                            
                            // Update the message content
                            const contentElement = responseElement.querySelector('.message-content');
                            contentElement.innerHTML = markdownToHtml(responseText);
                            
                            // Add syntax highlighting if needed
                            if (responseText.includes('```')) {
                                highlightCodeBlocks(contentElement);
                            }
                            
                            // Scroll to bottom
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        } else if (data.type === 'error') {
                            throw new Error(data.message);
                        } else if (data.type === 'done') {
                            // Add sources button if we have sources
                            if (lastSources && lastSources.length > 0) {
                                const sourcesButton = document.createElement('button');
                                sourcesButton.className = 'sources-btn';
                                sourcesButton.innerHTML = '<i class="fas fa-info-circle"></i> View Sources';
                                sourcesButton.addEventListener('click', showSources);
                                responseElement.appendChild(sourcesButton);
                            }
                            
                            // End of response
                            break;
                        }
                    } catch (error) {
                        console.error('Error parsing SSE message:', error);
                    }
                }
            }
        } catch (error) {
            console.error('Error fetching response:', error);
            
            // Remove typing indicator
            removeTypingIndicator(typingIndicatorId);
            
            // Add error message
            addMessage(`Sorry, there was an error processing your request: ${error.message}`, 'system');
        } finally {
            // Re-enable input
            isAwaitingResponse = false;
            userInput.disabled = false;
            userInput.focus();
        }
    }
    
    // Add a message to the chat
    function addMessage(content, type) {
        const messageElement = createMessageElement(content, type);
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Add to chat history
        chatHistory.push({
            content,
            type
        });
    }
    
    // Create a message element
    function createMessageElement(content, type) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${type}`;
        
        if (content) {
            const contentElement = document.createElement('div');
            contentElement.className = type === 'assistant' ? 'message-content markdown-content' : 'message-content';
            
            if (type === 'assistant') {
                contentElement.innerHTML = markdownToHtml(content);
                // Apply syntax highlighting to code blocks
                highlightCodeBlocks(contentElement);
            } else {
                contentElement.textContent = content;
            }
            
            messageElement.appendChild(contentElement);
        }
        
        return messageElement;
    }
    
    // Add typing indicator
    function addTypingIndicator() {
        const id = 'typing-indicator-' + Date.now();
        const typingElement = document.createElement('div');
        typingElement.id = id;
        typingElement.className = 'message assistant';
        
        const contentElement = document.createElement('div');
        contentElement.className = 'typing-indicator';
        contentElement.innerHTML = `
            <span></span>
            <span></span>
            <span></span>
        `;
        
        typingElement.appendChild(contentElement);
        chatMessages.appendChild(typingElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return id;
    }
    
    // Remove typing indicator
    function removeTypingIndicator(id) {
        const element = document.getElementById(id);
        if (element) element.remove();
    }
    
    // Apply syntax highlighting to code blocks
    function highlightCodeBlocks(element) {
        // This is a placeholder for syntax highlighting
        // You can integrate libraries like Prism.js or Highlight.js here
        const codeBlocks = element.querySelectorAll('pre code');
        if (codeBlocks.length > 0) {
            // If a syntax highlighting library is available, apply it here
            // For now, we'll just add a class
            codeBlocks.forEach(block => {
                block.classList.add('highlighted');
            });
        }
    }
    
    // Show sources modal
    function showSources() {
        const sourcesContent = document.getElementById('sources-content');
        
        if (!lastSources || lastSources.length === 0) {
            sourcesContent.innerHTML = '<div class="alert alert-info">No source information available.</div>';
            sourcesModal.show();
            return;
        }
        
        let html = '<div class="list-group">';
        
        lastSources.forEach((source, index) => {
            const scorePercentage = Math.round(source.score * 100);
            const scoreClass = scorePercentage > 75 ? 'bg-success' : 
                              scorePercentage > 50 ? 'bg-info' : 
                              scorePercentage > 25 ? 'bg-warning' : 'bg-danger';
            
            html += `
                <div class="source-item">
                    <div class="d-flex justify-content-between align-items-start">
                        <h6 class="source-title">${source.title || `Source ${index + 1}`}</h6>
                        <span class="source-score ${scoreClass} text-white">${scorePercentage}%</span>
                    </div>
                    <div class="source-meta">
                        <span class="badge bg-secondary">${source.type || 'Unknown'}</span>
                        ${source.url ? `<a href="${source.url}" target="_blank" class="ms-2"><i class="fas fa-external-link-alt"></i> View</a>` : ''}
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        sourcesContent.innerHTML = html;
        sourcesModal.show();
    }
    
    // Handle index refresh
    async function handleIndexRefresh() {
        btnIndex.disabled = true;
        btnIndex.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Refreshing...';
        
        // Get selected sources
        const sources = [];
        if (sourceConfluence.checked) sources.push('confluence');
        if (sourceJira.checked) sources.push('jira');
        if (sourceRemedy.checked) sources.push('remedy');
        
        try {
            const response = await fetch('/api/index', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    sources,
                    force_rebuild: true
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                indexToastBody.innerHTML = `<div class="text-success"><i class="fas fa-check-circle"></i> ${data.message}</div>`;
            } else {
                indexToastBody.innerHTML = `<div class="text-danger"><i class="fas fa-exclamation-circle"></i> ${data.error.message}</div>`;
            }
            
            indexToast.show();
        } catch (error) {
            console.error('Error refreshing index:', error);
            indexToastBody.innerHTML = `<div class="text-danger"><i class="fas fa-exclamation-circle"></i> Error refreshing index: ${error.message}</div>`;
            indexToast.show();
        } finally {
            btnIndex.disabled = false;
            btnIndex.innerHTML = '<i class="fas fa-sync"></i> Refresh Index';
        }
    }
});
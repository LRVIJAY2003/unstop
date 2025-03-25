"""
Logging utilities for the application.
"""
import logging
import os
import sys
from datetime import datetime

def setup_logger(log_level="INFO", log_file=None):
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Convert log level string to logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    return logger









"""
Caching utilities for the application.
"""
import os
import time
import json
import pickle
import hashlib
import logging
from functools import wraps
from typing import Any, Dict, Callable

logger = logging.getLogger(__name__)

def get_cache_key(func_name, args, kwargs):
    """Generate a cache key from function name and arguments.
    
    Args:
        func_name: Function name
        args: Function positional arguments
        kwargs: Function keyword arguments
        
    Returns:
        str: Cache key
    """
    # Convert arguments to string representation
    arg_str = str(args) + str(sorted(kwargs.items()))
    
    # Hash the argument string
    return f"{func_name}_{hashlib.md5(arg_str.encode()).hexdigest()}"

def cache_result(ttl=3600, cache_dir="cache"):
    """Decorator to cache function results.
    
    Args:
        ttl: Time-to-live in seconds (default: 1 hour)
        cache_dir: Directory to store cache files
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Generate cache key
            cache_key = get_cache_key(func.__name__, args, kwargs)
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Check if cache file exists and is not expired
            if os.path.exists(cache_file):
                # Get file modification time
                mod_time = os.path.getmtime(cache_file)
                
                # Check if cache is still valid
                if time.time() - mod_time < ttl:
                    try:
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        logger.debug(f"Cache hit for {func.__name__}")
                        return result
                    except Exception as e:
                        logger.warning(f"Error loading cache for {func.__name__}: {str(e)}")
            
            # Cache miss or error, call the function
            result = func(*args, **kwargs)
            
            # Save result to cache
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Cached result for {func.__name__}")
            except Exception as e:
                logger.warning(f"Error caching result for {func.__name__}: {str(e)}")
            
            return result
        
        return wrapper
    
    return decorator

class SimpleCache:
    """Simple in-memory cache with expiration."""
    
    def __init__(self, ttl=3600):
        """Initialize the cache.
        
        Args:
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key):
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        if key not in self.cache:
            return None
            
        value, timestamp = self.cache[key]
        
        # Check if value is expired
        if time.time() - timestamp > self.ttl:
            # Remove expired value
            del self.cache[key]
            return None
            
        return value
    
    def set(self, key, value):
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = (value, time.time())
    
    def delete(self, key):
        """Delete value from cache.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all values from cache."""
        self.cache.clear()
    
    def get_or_set(self, key, func):
        """Get value from cache or set it using function.
        
        Args:
            key: Cache key
            func: Function to call if cache miss
            
        Returns:
            Any: Cached or newly computed value
        """
        value = self.get(key)
        
        if value is None:
            # Cache miss, call function
            value = func()
            self.set(key, value)
            
        return value







"""
Text processing utilities for the application.
"""
import re
import nltk
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get stopwords
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

def preprocess_text(text: str) -> str:
    """Preprocess text for analysis.
    
    Args:
        text: Input text
        
    Returns:
        str: Preprocessed text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words.
    
    Args:
        text: Input text
        
    Returns:
        list: List of tokens
    """
    if not text:
        return []
        
    # Preprocess text
    text = preprocess_text(text)
    
    # Tokenize
    tokens = text.split()
    
    return tokens

def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove stopwords from tokens.
    
    Args:
        tokens: List of tokens
        
    Returns:
        list: Filtered tokens
    """
    return [token for token in tokens if token not in STOPWORDS]

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens.
    
    Args:
        tokens: List of tokens
        
    Returns:
        list: Lemmatized tokens
    """
    return [lemmatizer.lemmatize(token) for token in tokens]

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract keywords from text.
    
    Args:
        text: Input text
        top_n: Number of keywords to extract
        
    Returns:
        list: List of keywords
    """
    if not text:
        return []
        
    # Tokenize
    tokens = tokenize_text(text)
    
    # Remove stopwords
    filtered_tokens = remove_stopwords(tokens)
    
    # Lemmatize
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)
    
    # Count token frequencies
    token_counts = {}
    for token in lemmatized_tokens:
        if len(token) > 2:  # Only consider tokens with length > 2
            token_counts[token] = token_counts.get(token, 0) + 1
    
    # Sort by frequency (descending)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get top-n keywords
    keywords = [token for token, _ in sorted_tokens[:top_n]]
    
    return keywords

def extract_named_entities(text: str) -> List[Tuple[str, str]]:
    """Extract named entities from text.
    
    Args:
        text: Input text
        
    Returns:
        list: List of (entity, entity_type) tuples
    """
    if not text:
        return []
        
    # Simple pattern-based entity extraction (as fallback)
    entities = []
    
    # Extract dates
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY or DD-MM-YYYY
        r'\d{4}-\d{2}-\d{2}'         # YYYY-MM-DD (ISO format)
    ]
    
    for pattern in date_patterns:
        for match in re.finditer(pattern, text):
            entities.append((match.group(0), 'DATE'))
    
    # Extract emails
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_pattern, text):
        entities.append((match.group(0), 'EMAIL'))
    
    # Extract URLs
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    for match in re.finditer(url_pattern, text):
        entities.append((match.group(0), 'URL'))
    
    return entities

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        list: List of sentences
    """
    if not text:
        return []
        
    try:
        return nltk.sent_tokenize(text)
    except:
        # Fallback to simple sentence splitting
        return re.split(r'(?<=[.!?])\s+', text)

def clean_text(text: str) -> str:
    """Clean text by removing special characters and normalizing whitespace.
    
    Args:
        text: Input text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
        
    # Replace special characters
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text












"""
HTML parsing utilities for the application.
"""
import re
import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def extract_text_from_html(html: str) -> str:
    """Extract plain text from HTML.
    
    Args:
        html: HTML content
        
    Returns:
        str: Plain text
    """
    if not html:
        return ""
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text(separator=" ")
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}")
        return html  # Return original HTML if extraction fails

def extract_links_from_html(html: str) -> List[Dict[str, str]]:
    """Extract links from HTML.
    
    Args:
        html: HTML content
        
    Returns:
        list: List of dictionaries with link information
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract links
        links = []
        for a in soup.find_all('a', href=True):
            text = a.get_text().strip()
            href = a['href']
            
            links.append({
                'text': text,
                'url': href
            })
            
        return links
        
    except Exception as e:
        logger.error(f"Error extracting links from HTML: {str(e)}")
        return []

def extract_images_from_html(html: str) -> List[Dict[str, str]]:
    """Extract images from HTML.
    
    Args:
        html: HTML content
        
    Returns:
        list: List of dictionaries with image information
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract images
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            
            if src:
                images.append({
                    'src': src,
                    'alt': alt
                })
            
        return images
        
    except Exception as e:
        logger.error(f"Error extracting images from HTML: {str(e)}")
        return []

def extract_tables_from_html(html: str) -> List[List[List[str]]]:
    """Extract tables from HTML.
    
    Args:
        html: HTML content
        
    Returns:
        list: List of tables, where each table is a list of rows and each row is a list of cells
    """
    if not html:
        return []
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract tables
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            
            # Process rows
            for tr in table.find_all('tr'):
                row_data = []
                
                # Process cells
                for td in tr.find_all(['td', 'th']):
                    cell_text = td.get_text().strip()
                    row_data.append(cell_text)
                
                if row_data:
                    table_data.append(row_data)
            
            if table_data:
                tables.append(table_data)
            
        return tables
        
    except Exception as e:
        logger.error(f"Error extracting tables from HTML: {str(e)}")
        return []

def html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown.
    
    Args:
        html: HTML content
        
    Returns:
        str: Markdown text
    """
    if not html:
        return ""
        
    try:
        import html2text
        
        # Create converter
        h2t = html2text.HTML2Text()
        h2t.ignore_links = False
        h2t.ignore_images = False
        h2t.ignore_tables = False
        
        # Convert HTML to Markdown
        markdown = h2t.handle(html)
        
        return markdown
        
    except ImportError:
        logger.warning("html2text not available, falling back to simple text extraction")
        return extract_text_from_html(html)
    except Exception as e:
        logger.error(f"Error converting HTML to Markdown: {str(e)}")
        return extract_text_from_html(html)

def clean_html(html: str) -> str:
    """Clean HTML by removing unnecessary elements and attributes.
    
    Args:
        html: HTML content
        
    Returns:
        str: Cleaned HTML
    """
    if not html:
        return ""
        
    try:
        # Parse HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.extract()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Return cleaned HTML
        return str(soup)
        
    except Exception as e:
        logger.error(f"Error cleaning HTML: {str(e)}")
        return html  # Return original HTML if cleaning fails











"""
Input validation utilities for the application.
"""
import re
import logging
from typing import Any, Dict, List, Union, Optional

logger = logging.getLogger(__name__)

def validate_query(query: str) -> bool:
    """Validate a query string.
    
    Args:
        query: Query string
        
    Returns:
        bool: True if query is valid, False otherwise
    """
    if not query:
        return False
        
    # Check if query is too short
    if len(query.strip()) < 3:
        return False
        
    # Check if query is too long
    if len(query) > 1000:
        return False
        
    return True

def validate_url(url: str) -> bool:
    """Validate a URL.
    
    Args:
        url: URL string
        
    Returns:
        bool: True if URL is valid, False otherwise
    """
    if not url:
        return False
        
    # Check if URL is well-formed
    url_pattern = r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    if not re.match(url_pattern, url):
        return False
        
    return True

def validate_email(email: str) -> bool:
    """Validate an email address.
    
    Args:
        email: Email string
        
    Returns:
        bool: True if email is valid, False otherwise
    """
    if not email:
        return False
        
    # Check if email is well-formed
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False
        
    return True

def validate_api_token(token: str) -> bool:
    """Validate an API token.
    
    Args:
        token: API token string
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    if not token:
        return False
        
    # Check if token is too short
    if len(token) < 10:
        return False
        
    return True

def validate_request_data(data: Dict[str, Any], required_fields: List[str]) -> Dict[str, str]:
    """Validate request data.
    
    Args:
        data: Request data
        required_fields: List of required field names
        
    Returns:
        dict: Dictionary of validation errors (field name -> error message)
    """
    errors = {}
    
    if not data:
        return {'general': 'No data provided'}
        
    # Check required fields
    for field in required_fields:
        if field not in data or not data[field]:
            errors[field] = f'Field "{field}" is required'
    
    return errors

def sanitize_input(input_str: str) -> str:
    """Sanitize input string.
    
    Args:
        input_str: Input string
        
    Returns:
        str: Sanitized string
    """
    if not input_str:
        return ""
        
    # Remove potential script tags
    sanitized = re.sub(r'<script.*?>.*?</script>', '', input_str, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove potential event handlers
    sanitized = re.sub(r' on\w+=".*?"', ' ', sanitized, flags=re.IGNORECASE)
    
    # Remove potential javascript: URLs
    sanitized = re.sub(r'javascript:', 'blocked:', sanitized, flags=re.IGNORECASE)
    
    return sanitized








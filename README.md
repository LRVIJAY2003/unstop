"""
Configuration management for the application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class."""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'
    
    # Confluence settings
    CONFLUENCE_URL = os.environ.get('CONFLUENCE_URL')
    CONFLUENCE_SPACE_ID = os.environ.get('CONFLUENCE_SPACE_ID')
    CONFLUENCE_USER_ID = os.environ.get('CONFLUENCE_USER_ID')
    CONFLUENCE_API_TOKEN = os.environ.get('CONFLUENCE_API_TOKEN')
    
    # Remedy settings
    REMEDY_SERVER = os.environ.get('REMEDY_SERVER')
    REMEDY_API_BASE = os.environ.get('REMEDY_API_BASE')
    REMEDY_USERNAME = os.environ.get('REMEDY_USERNAME')
    REMEDY_PASSWORD = os.environ.get('REMEDY_PASSWORD')
    
    # Google AI settings
    PROJECT_ID = os.environ.get('PROJECT_ID')
    REGION = os.environ.get('REGION')
    MODEL_NAME = os.environ.get('MODEL_NAME', 'gemini-1.0-flash-001')
    
    # Application settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    CACHE_DIR = os.environ.get('CACHE_DIR', 'cache')
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '100'))
    
    @staticmethod
    def init_app(app):
        """Initialize app with config."""
        # Create cache directory if it doesn't exist
        os.makedirs(Config.CACHE_DIR, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Get current configuration
def get_config():
    """Return the current configuration based on FLASK_ENV."""
    config_name = os.environ.get('FLASK_ENV', 'development')
    return config.get(config_name, config['default'])
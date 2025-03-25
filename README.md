"""
Client for interacting with Confluence.
"""
import logging
import requests
from atlassian import Confluence
from bs4 import BeautifulSoup
import html2text

from src.confluence.parser import parse_confluence_content
from src.utils.cache import cache_result

logger = logging.getLogger(__name__)

class ConfluenceClient:
    """Client for interacting with Confluence API."""
    
    def __init__(self, url, username, api_token, space_id):
        """Initialize the Confluence client.
        
        Args:
            url: Confluence URL
            username: Confluence username/email
            api_token: Confluence API token
            space_id: Confluence space ID
        """
        self.url = url
        self.username = username
        self.api_token = api_token
        self.space_id = space_id
        
        self.client = Confluence(
            url=url,
            username=username,
            password=api_token,
            cloud=True  # Set to False if using server installation
        )
        
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = False
        self.h2t.ignore_tables = False
        
    def check_connection(self):
        """Check if the connection to Confluence is working.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            self.client.get_space(self.space_id)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Confluence: {str(e)}")
            return False
    
    @cache_result(ttl=3600)  # Cache for 1 hour
    def search_content(self, query, limit=5):
        """Search for content in Confluence.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            list: List of dictionaries containing page info and content
        """
        try:
            # Search for pages matching the query
            cql = f'space="{self.space_id}" AND text ~ "{query}"'
            search_results = self.client.cql(cql, limit=limit)
            
            results = []
            for result in search_results.get('results', []):
                page_id = result.get('content', {}).get('id')
                if not page_id:
                    continue
                
                # Get page content
                page = self.client.get_page_by_id(
                    page_id, 
                    expand='body.storage,version'
                )
                
                # Extract and parse HTML content
                html_content = page.get('body', {}).get('storage', {}).get('value', '')
                text_content = self.h2t.handle(html_content)
                
                # Parse content with tables, lists, etc.
                parsed_content = parse_confluence_content(html_content)
                
                results.append({
                    'id': page_id,
                    'title': page.get('title', ''),
                    'url': f"{self.url}/wiki/spaces/{self.space_id}/pages/{page_id}",
                    'content': parsed_content,
                    'text_content': text_content,
                    'last_updated': page.get('version', {}).get('when', '')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching Confluence: {str(e)}")
            raise
    
    def get_page_attachments(self, page_id):
        """Get attachments for a page.
        
        Args:
            page_id: Confluence page ID
            
        Returns:
            list: List of attachment metadata
        """
        try:
            attachments = self.client.get_attachments_from_content(page_id)
            return attachments.get('results', [])
        except Exception as e:
            logger.error(f"Error getting page attachments: {str(e)}")
            return []
    
    def get_attachment_content(self, page_id, attachment_id, filename):
        """Get attachment content.
        
        Args:
            page_id: Confluence page ID
            attachment_id: Attachment ID
            filename: Attachment filename
            
        Returns:
            bytes: Attachment content as bytes
        """
        try:
            return self.client.download_attachment(
                page_id, 
                attachment_id, 
                filename
            )
        except Exception as e:
            logger.error(f"Error downloading attachment: {str(e)}")
            return None
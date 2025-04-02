"""
Confluence API client for the RAG system.
"""
import requests
import logging
import json
import os
import sys
import urllib3
from html.parser import HTMLParser
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

class HTMLFilter(HTMLParser):
    """
    Filter to extract text from HTML content.
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
    
    def __init__(self, base_url=None, username=None, api_token=None, ssl_verify=False, ca_bundle=None):
        """
        Initialize the Confluence client with authentication details and SSL options.
        
        Args:
            base_url: The base URL of the Jira instance (e.g., https://cmegroup.atlassian.net)
            username: The email address for authentication
            api_token: The API token for authentication
            ssl_verify: Whether to verify SSL certificates (set to False to disable verification)
            ca_bundle: Path to a CA bundle file to use for verification (alternative to disabling verification)
        """
        self.base_url = base_url or Config.CONFLUENCE_URL
        self.username = username or Config.CONFLUENCE_USERNAME
        self.api_token = api_token or Config.CONFLUENCE_API_TOKEN
        
        # Handle SSL verification
        if ssl_verify is False:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled. This is not recommended for production use.")
        
        self.ssl_verify = ssl_verify
        self.ca_bundle = ca_bundle
        
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
            tuple: (success_flag, server_info)
        """
        try:
            logger.info("Testing connection to Confluence...")
            response = requests.get(
                f"{self.base_url}/rest/api/2/serverInfo",
                auth=(self.username, self.api_token),
                headers=self.headers,
                verify=self.ssl_verify
            )
            
            if response.status_code == 200:
                server_info = response.json()
                server_version = server_info.get('version', 'Unknown')
                logger.info(f"Connection to Confluence successful! Version: {server_version}")
                return True, server_version
            else:
                logger.error(f"Connection test failed: {response.status_code}")
                if hasattr(response, 'text'):
                    logger.error(f"Response: {response.text}")
                return False, f"Status code: {response.status_code}"
        except requests.RequestException as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False, str(e)
    
    def get_content_by_id(self, content_id, expand=None):
        """
        Get a specific content by its ID.
        
        Args:
            content_id: The ID of the content
            expand: Optional comma-separated list of properties to expand
            
        Returns:
            dict: Content data or None if not found/error
        """
        try:
            # Build expand parameter
            expand_param = expand if expand else "body.storage,metadata.labels"
            
            logger.info(f"Fetching content: {content_id}")
            response = requests.get(
                f"{self.base_url}/rest/api/content/{content_id}",
                auth=(self.username, self.api_token),
                headers=self.headers,
                params={"expand": expand_param},
                verify=self.ssl_verify
            )
            
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            content = response.json()
            logger.info(f"Successfully retrieved content: {content_id}")
            return content
        except requests.RequestException as e:
            logger.error(f"Error getting content {content_id}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                try:
                    error_details = e.response.json()
                    logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
                except ValueError:
                    logger.error(f"Response content: {e.response.text}")
            return None
    
    def search_content(self, cql=None, title=None, content_type="page", expand=None, limit=25, start=0):
        """
        Search for content using CQL or specific parameters.
        
        Args:
            cql: Confluence Query Language string
            title: Title to search for
            content_type: Type of content to search for (default: page)
            expand: Optional comma-separated list of properties to expand
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
                params["expand"] = expand
            
            logger.info(f"Searching for content with params: {params}")
            response = requests.get(
                f"{self.base_url}/rest/api/content/search",
                auth=(self.username, self.api_token),
                headers=self.headers,
                params=params,
                verify=self.ssl_verify
            )
            
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            results = response.json()
            logger.info(f"Search returned {len(results.get('results', []))} results")
            return results.get('results', [])
        except requests.RequestException as e:
            logger.error(f"Error searching content: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Status code: {e.response.status_code}")
                logger.error(f"Response: {e.response.text}")
            return []
    
    def get_all_content(self, content_type="page", limit=25):
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
                
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                data = response.json()
                results = data.get('results', [])
                all_content.extend(results)
                
                # Check if there are more pages
                if len(results) < limit:
                    break
                
                # Move to next page
                start += limit
                logger.info(f"Retrieved {len(all_content)} {content_type} items so far")
            except requests.RequestException as e:
                logger.error(f"Error retrieving all content: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Status code: {e.response.status_code}")
                    logger.error(f"Response: {e.response.text}")
                break
        
        logger.info(f"Retrieved a total of {len(all_content)} {content_type} items")
        return all_content
    
    def get_page_content(self, page_id):
        """
        Get the content of a page in a suitable format for RAG.
        This extracts and processes the content to be more suitable for embeddings.
        
        Args:
            page_id: The ID of the page
            
        Returns:
            dict: Processed page content or None
        """
        try:
            # Get page with body.storage and metadata
            page = self.get_content_by_id(page_id, expand="body.storage,metadata.labels")
            
            if not page:
                return None
            
            # Extract basic metadata
            metadata = {
                "id": page.get("id"),
                "title": page.get("title"),
                "type": page.get("type"),
                "url": f"{self.base_url}/wiki{page.get('_links', {}).get('webui', '')}",
                "created": page.get("created"),
                "updated": page.get("history", {}).get("modified")
            }
            
            # Add labels if available
            if "metadata" in page and "labels" in page["metadata"]:
                metadata["labels"] = [label.get("name") for label in page["metadata"]["labels"].get("results", [])]
            else:
                metadata["labels"] = []
            
            # Get the content
            body = page.get("body", {}).get("storage", {}).get("value", "")
            
            # Convert HTML to plain text
            html_parser = HTMLFilter()
            html_parser.feed(body)
            plain_text = html_parser.text
            
            # Return structured content
            return {
                "metadata": metadata,
                "content": plain_text,
                "raw_html": body
            }
        except Exception as e:
            logger.error(f"Error processing page content: {str(e)}")
            return None
    
    def get_all_pages_for_rag(self, limit=100):
        """
        Get all pages and process them for RAG.
        
        Args:
            limit: Maximum number of pages to retrieve
            
        Returns:
            list: List of processed pages for RAG
        """
        try:
            # Get all page IDs
            pages = self.get_all_content(content_type="page", limit=limit)
            page_ids = [page.get("id") for page in pages]
            
            # Process each page
            processed_pages = []
            for page_id in page_ids:
                processed_page = self.get_page_content(page_id)
                if processed_page:
                    processed_pages.append(processed_page)
            
            logger.info(f"Processed {len(processed_pages)} pages for RAG")
            return processed_pages
        except Exception as e:
            logger.error(f"Error getting pages for RAG: {str(e)}")
            return []

# Create global instance
confluence_client = ConfluenceClient()

















"""
Confluence connector for the RAG system.
"""
import logging
from typing import List, Dict, Any, Optional
from data_sources.confluence.client import confluence_client

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
    
    def get_spaces(self) -> List[Dict[str, Any]]:
        """
        Get all Confluence spaces.
        
        Returns:
            List of spaces with key information
        """
        try:
            # This is a placeholder, as spaces aren't directly exposed in your client code
            # You would need to implement a proper spaces fetch API call based on your specific endpoint
            return []
        except Exception as e:
            logger.error(f"Error getting spaces: {str(e)}")
            return []
    
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
            page_contents = self.client.get_all_pages_for_rag(limit=limit)
            
            # Format for RAG
            rag_documents = []
            for page in page_contents:
                if not page:
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

# Global instance
confluence_connector = ConfluenceConnector()
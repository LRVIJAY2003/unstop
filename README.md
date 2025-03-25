"""
Client for interacting with BMC Remedy.
"""
import logging
import json
import requests
from requests.auth import HTTPBasicAuth

from src.utils.cache import cache_result

logger = logging.getLogger(__name__)

class RemedyClient:
    """Client for interacting with Remedy API."""
    
    def __init__(self, server, api_base, username, password):
        """Initialize the Remedy client.
        
        Args:
            server: Remedy server hostname
            api_base: Base URL for Remedy API
            username: Remedy username
            password: Remedy password
        """
        self.server = server
        self.api_base = api_base
        self.username = username
        self.password = password
        
        # API endpoints
        self.login_url = f"{api_base}/api/jwt/login"
        self.logout_url = f"{api_base}/api/jwt/logout"
        self.ticket_url = f"{api_base}/api/arsys/v1/entry/HPD:Help"
        
        # Authentication token
        self.token = None
        self.logged_in = False
    
    def _do_login(self):
        """Login to Remedy API and get authentication token."""
        if self.logged_in:
            return True
            
        try:
            # Prepare login payload
            payload = {
                'username': self.username,
                'password': self.password
            }
            
            # Set request headers
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            # Make login request
            response = requests.post(
                self.login_url,
                data=payload,
                headers=headers,
                verify=False  # TODO: Enable verification with proper certificates
            )
            
            # Check if login was successful
            if response.status_code == 200:
                self.token = response.text
                self.logged_in = True
                logger.info("Successfully logged in to Remedy API")
                return True
            else:
                logger.error(f"Failed to login to Remedy: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error logging in to Remedy: {str(e)}")
            return False
    
    def _logout(self):
        """Logout from Remedy API."""
        if not self.logged_in or not self.token:
            return True
            
        try:
            # Set request headers
            headers = {
                'Authorization': f"AR-JWT {self.token}"
            }
            
            # Make logout request
            response = requests.post(
                self.logout_url,
                headers=headers,
                verify=False  # TODO: Enable verification with proper certificates
            )
            
            # Check if logout was successful
            if response.status_code == 204:
                self.token = None
                self.logged_in = False
                logger.info("Successfully logged out from Remedy API")
                return True
            else:
                logger.error(f"Failed to logout from Remedy: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error logging out from Remedy: {str(e)}")
            return False
    
    def check_connection(self):
        """Check if the connection to Remedy is working.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            return self._do_login() and self._logout()
        except Exception as e:
            logger.error(f"Failed to connect to Remedy: {str(e)}")
            return False
    
    @cache_result(ttl=300)  # Cache for 5 minutes
    def search_tickets(self, query, limit=10):
        """Search for tickets in Remedy.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            list: List of dictionaries containing ticket information
        """
        try:
            # Login to get token
            if not self._do_login():
                logger.error("Failed to login to Remedy")
                return []
                
            # Create authentication headers
            headers = {
                'Authorization': f"AR-JWT {self.token}",
                'Content-Type': 'application/json'
            }
            
            # Create query parameters
            # Using a simple query to get tickets with terms in Status or Description
            query_params = {
                'q': f"'Description' LIKE \"%{query}%\"",
                'limit': limit,
                'fields': 'values(ID, Status, Description, Submitter, "Request Assignee", "Impact", Priority)'
            }
            
            # Make search request
            response = requests.get(
                self.ticket_url,
                headers=headers,
                params=query_params,
                verify=False  # TODO: Enable verification with proper certificates
            )
            
            # Check if search was successful
            if response.status_code == 200:
                data = response.json()
                entries = data.get('entries', [])
                
                # Format ticket data
                tickets = []
                for entry in entries:
                    values = entry.get('values', {})
                    tickets.append({
                        'id': values.get('ID', ''),
                        'status': values.get('Status', ''),
                        'description': values.get('Description', ''),
                        'submitter': values.get('Submitter', ''),
                        'assignee': values.get('Request Assignee', ''),
                        'impact': values.get('Impact', ''),
                        'priority': values.get('Priority', '')
                    })
                
                # Logout to clean up
                self._logout()
                
                return tickets
            else:
                logger.error(f"Failed to search Remedy tickets: {response.status_code} - {response.text}")
                # Logout to clean up
                self._logout()
                return []
                
        except Exception as e:
            logger.error(f"Error searching Remedy tickets: {str(e)}")
            # Try to logout anyway
            try:
                self._logout()
            except:
                pass
            return []
    
    def get_ticket_details(self, ticket_id):
        """Get detailed information for a specific ticket.
        
        Args:
            ticket_id: Ticket ID
            
        Returns:
            dict: Dictionary containing ticket details
        """
        try:
            # Login to get token
            if not self._do_login():
                logger.error("Failed to login to Remedy")
                return {}
                
            # Create authentication headers
            headers = {
                'Authorization': f"AR-JWT {self.token}",
                'Content-Type': 'application/json'
            }
            
            # Make request to get ticket details
            response = requests.get(
                f"{self.ticket_url}/{ticket_id}",
                headers=headers,
                verify=False  # TODO: Enable verification with proper certificates
            )
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                values = data.get('values', {})
                
                # Format ticket details
                ticket_details = {
                    'id': values.get('Incident Number', ''),
                    'status': values.get('Status', ''),
                    'description': values.get('Description', ''),
                    'submitter': values.get('Submitter', ''),
                    'assignee': values.get('Assignee', ''),
                    'assignee_group': values.get('Assignee Group', ''),
                    'impact': values.get('Impact', ''),
                    'priority': values.get('Priority', ''),
                    'urgency': values.get('Urgency', ''),
                    'reported_source': values.get('Reported Source', ''),
                    'last_modified': values.get('Last Modified Date', ''),
                    'create_date': values.get('Create Date', ''),
                    'resolution': values.get('Resolution', '')
                }
                
                # Logout to clean up
                self._logout()
                
                return ticket_details
            else:
                logger.error(f"Failed to get ticket details: {response.status_code} - {response.text}")
                # Logout to clean up
                self._logout()
                return {}
                
        except Exception as e:
            logger.error(f"Error getting ticket details: {str(e)}")
            # Try to logout anyway
            try:
                self._logout()
            except:
                pass
            return {}
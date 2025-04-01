# Replace the login method with your original working code
def login(self):
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
    
    # Add direct connection pooling settings to avoid deadlocks
    try:
        # Using session for better connection handling
        session = requests.Session()
        r = session.post(url, data=payload, headers=headers, verify=self.ssl_verify, 
                        timeout=30)  # Add timeout
        
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










# Add after the app creation in app.py
@app.before_first_request
def test_connections():
    logger.info("Testing data source connections...")
    
    # Test Confluence
    try:
        success = confluence_connector.test_connection()
        logger.info(f"Confluence connection test: {'Success' if success else 'Failed'}")
    except Exception as e:
        logger.error(f"Confluence connection error: {str(e)}")
    
    # Test JIRA
    try:
        success = jira_connector.test_connection()
        logger.info(f"JIRA connection test: {'Success' if success else 'Failed'}")
    except Exception as e:
        logger.error(f"JIRA connection error: {str(e)}")





from app import app

if __name__ == '__main__':
    app.config['DEBUG'] = True
    app.run(host='0.0.0.0', port=5000, use_reloader=True)







import json
import requests
import logging
import urllib3
import getpass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from urllib.parse import quote

from config import REMEDY_URL, REMEDY_USERNAME, REMEDY_PASSWORD, REMEDY_SSL_VERIFY
from utils.logger import setup_module_logger

logger = setup_module_logger("remedy_client")

class RemedyClient:
    """
    Client for BMC Remedy REST API operations with comprehensive error handling and
    advanced querying.
    """
    
    def __init__(self, server_url=None, username=None, password=None, ssl_verify=None):
        """
        Initialize the Remedy client with server and authentication details.
        
        Args:
            server_url: The base URL of the Remedy server (e.g., https://cmegroup-restapi.onbmc.com)
            username: Username for authentication (will prompt if None)
            password: Password for authentication (will prompt if None)
            ssl_verify: Whether to verify SSL certificates (set to False to disable verification)
        """
        self.server_url = server_url or REMEDY_URL
        self.server_url = self.server_url.rstrip('/')  # Remove trailing slash if present
        self.username = username or REMEDY_USERNAME
        self.password = password or REMEDY_PASSWORD
        self.token = None
        self.token_type = "AR-JWT"
        
        # Handle SSL verification
        if ssl_verify is None:
            ssl_verify = REMEDY_SSL_VERIFY
            
        if ssl_verify is False:
            # Disable SSL warnings if verification is disabled
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warning("SSL certificate verification is disabled.")
        self.ssl_verify = ssl_verify
        
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
                print(f"Failure...")
                print(f"Status Code: {r.status_code}")
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
    
    def get_incident(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific incident by its ID.
        
        Args:
            incident_id: The Incident Number (e.g., INC000001482087)
            
        Returns:
            dict: Incident data or None if not found/error
        """
        if not self.token:
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
        if not self.token:
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
    
    def get_incidents_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get incidents by their status.
        
        Args:
            status: The status to filter by (e.g., "Open", "Closed", "Resolved")
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
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
    
    def get_incidents_by_assignee(self, assignee: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get incidents assigned to a specific person.
        
        Args:
            assignee: The assignee name
            limit: Maximum number of incidents to retrieve
            
        Returns:
            list: List of incidents or empty list if none found/error
        """
        if not self.token:
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
        if not self.token:
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
    
    def create_incident(self, summary: str, description: str, impact: str = "4-Minor/Localized", urgency: str = "4-Low",
                       reported_source: str = "Direct Input", service_type: str = "User Service Restoration",
                       assigned_group: str = None) -> Optional[Dict[str, Any]]:
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
        if not self.token:
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
        if not self.token:
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
        if not self.token:
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
        content_parts
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
        
        return {
            "metadata": metadata,
            "content": content,
            
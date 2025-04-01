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
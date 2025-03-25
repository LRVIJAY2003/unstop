"""
WSGI entry point for gunicorn or similar WSGI servers.
"""
from app import create_app

application = create_app()

if __name__ == "__main__":
    application.run()
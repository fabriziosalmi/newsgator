"""
Web Server Module

This module provides a simple HTTP server to serve the generated HTML content and RSS feed.
"""

import logging
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path

from newsgator.config import DOCS_DIR, WEB_SERVER_PORT

logger = logging.getLogger(__name__)

class DocsDirectoryHandler(SimpleHTTPRequestHandler):
    """HTTP request handler that serves files from the docs directory."""
    
    def __init__(self, docs_directory, *args, **kwargs):
        self.docs_directory = docs_directory
        super().__init__(*args, **kwargs)
    
    def translate_path(self, path):
        """Translate URL path to file system path."""
        # First, use the parent class method to get a clean path
        path = super().translate_path(path)
        
        # Replace the current directory with the docs directory
        rel_path = os.path.relpath(path, os.getcwd())
        return os.path.join(self.docs_directory, rel_path)


def run_server(docs_directory=None, port=None):
    """
    Run a simple HTTP server to serve the generated content.
    
    Args:
        docs_directory: Directory containing the docs to serve.
        port: Port to run the server on.
    """
    docs_dir = Path(docs_directory or DOCS_DIR)
    server_port = port or WEB_SERVER_PORT
    
    if not docs_dir.exists():
        logger.error(f"Docs directory not found: {docs_dir}")
        return False
    
    handler = partial(DocsDirectoryHandler, str(docs_dir.resolve()))
    
    try:
        server = HTTPServer(('0.0.0.0', server_port), handler)
        logger.info(f"Starting HTTP server on port {server_port}, serving content from {docs_dir}")
        logger.info(f"Open http://localhost:{server_port} in your browser to view the news")
        server.serve_forever()
        return True
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return True
    except Exception as e:
        logger.error(f"Error running HTTP server: {str(e)}")
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get port from command line argument if provided
    port = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    # Run the server
    run_server(port=port)
"""
Web Server Module

This module provides a simple HTTP server to serve the generated HTML content and RSS feed.
"""

import logging
import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from rich.console import Console
from rich.logging import RichHandler

from newsgator.config import DOCS_DIR, WEB_SERVER_PORT

# Initialize Rich console for the web server
console = Console()

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
    
    def log_message(self, format, *args):
        """Override the default log_message method to use Rich logging."""
        status_code = args[1]
        path = args[0].split()[1]
        
        # Color-code status codes
        if status_code.startswith('2'):  # 2xx Success
            status_style = "[bold green]"
        elif status_code.startswith('3'):  # 3xx Redirection
            status_style = "[bold blue]"
        elif status_code.startswith('4'):  # 4xx Client Error
            status_style = "[bold yellow]"
        elif status_code.startswith('5'):  # 5xx Server Error
            status_style = "[bold red]"
        else:
            status_style = "[bold]"
        
        # Log with rich formatting
        console.log(f"[cyan]{self.client_address[0]}[/cyan] - {path} - {status_style}{status_code}[/]")
    
    def do_GET(self):
        """Handle GET requests, including API endpoints."""
        parsed_path = urlparse(self.path)
        
        # Handle API requests
        if parsed_path.path.startswith('/api/'):
            return self.handle_api_request(parsed_path)
            
        # Default behavior for non-API requests
        return super().do_GET()
    
    def handle_api_request(self, parsed_path):
        """Handle API requests."""
        # API endpoint to get all news
        if parsed_path.path == '/api/news':
            return self.send_api_response(self.get_all_news())
            
        # API endpoint to get news by topic
        elif parsed_path.path == '/api/news/topic':
            query_params = parse_qs(parsed_path.query)
            topic = query_params.get('q', [''])[0]
            if topic:
                return self.send_api_response(self.get_news_by_topic(topic))
            else:
                return self.send_error(400, "Missing 'q' parameter for topic search")
                
        # API endpoint to get news by language
        elif parsed_path.path == '/api/news/language':
            query_params = parse_qs(parsed_path.query)
            language = query_params.get('lang', [''])[0]
            if language:
                return self.send_api_response(self.get_news_by_language(language))
            else:
                return self.send_error(400, "Missing 'lang' parameter for language filter")
        
        # API endpoint to get news topics (list of all topics)
        elif parsed_path.path == '/api/topics':
            return self.send_api_response(self.get_all_topics())
        
        # API endpoint not found
        else:
            return self.send_error(404, "API endpoint not found")
    
    def send_api_response(self, data):
        """Send a JSON response for API requests."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')  # Enable CORS
        self.end_headers()
        
        # Convert data to JSON and send
        response = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.wfile.write(response)
        return
    
    def get_all_news(self):
        """Get all news articles."""
        try:
            articles_path = os.path.join(self.docs_directory, 'data', 'articles.json')
            with open(articles_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading news data: {str(e)}")
            return {"error": "Failed to load news data"}
    
    def get_news_by_topic(self, topic):
        """Get news articles filtered by topic."""
        all_news = self.get_all_news()
        
        if isinstance(all_news, dict) and "error" in all_news:
            return all_news
            
        # Search for topic (case-insensitive partial match)
        topic_lower = topic.lower()
        matching_topics = [
            item for item in all_news 
            if topic_lower in item.get("topic", "").lower()
        ]
        
        return matching_topics
    
    def get_news_by_language(self, language):
        """Get news articles filtered by language."""
        all_news = self.get_all_news()
        
        if isinstance(all_news, dict) and "error" in all_news:
            return all_news
            
        # Filter by language code
        result = []
        for topic in all_news:
            matching_articles = []
            for article in topic.get("articles", []):
                if article.get("language", "") == language:
                    matching_articles.append(article)
            
            if matching_articles:
                # Create a new topic entry with only the matching articles
                filtered_topic = topic.copy()
                filtered_topic["articles"] = matching_articles
                result.append(filtered_topic)
                
        return result
    
    def get_all_topics(self):
        """Get a list of all news topics."""
        all_news = self.get_all_news()
        
        if isinstance(all_news, dict) and "error" in all_news:
            return all_news
            
        # Extract just the topics
        topics = [{"topic": item.get("topic")} for item in all_news if "topic" in item]
        return topics

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
        logger.info(f"API available at http://localhost:{server_port}/api/news")
        server.serve_forever()
        return True
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return True
    except Exception as e:
        logger.error(f"Error running HTTP server: {str(e)}")
        return False


if __name__ == "__main__":
    # Setup logging with Rich
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=True, console=console, show_time=True, show_path=False)
        ]
    )
    
    # Get port from command line argument if provided
    port = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    # Run the server
    run_server(port=port)
"""
Newsgator Main Module

This is the main module for the Newsgator application, which orchestrates
the entire process of fetching, analyzing, processing, and publishing news content.
"""

import logging
import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple

from newsgator.feed_processing.feed_processor import FeedProcessor
from newsgator.content_analysis.content_analyzer import ContentAnalyzer
from newsgator.llm_integration.llm_processor import LLMProcessor
from newsgator.html_generation.html_generator import HTMLGenerator
from newsgator.config import DOCS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('newsgator.log')
    ]
)

logger = logging.getLogger(__name__)

class Newsgator:
    """Main class for the Newsgator application."""
    
    def __init__(self):
        """Initialize the Newsgator application."""
        self.feed_processor = FeedProcessor()
        self.content_analyzer = ContentAnalyzer()
        self.llm_processor = LLMProcessor()
        self.html_generator = HTMLGenerator()
    
    def run(self) -> bool:
        """
        Run the complete Newsgator workflow.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            logger.info("Starting Newsgator workflow")
            
            # 1. Fetch RSS feeds
            logger.info("Fetching RSS feeds")
            articles = self.feed_processor.fetch_all_feeds()
            
            if not articles:
                logger.error("No articles fetched, aborting")
                return False
            
            logger.info(f"Fetched {len(articles)} articles")
            
            # 2. Analyze and cluster articles
            logger.info("Analyzing and clustering articles")
            clusters = self.content_analyzer.cluster_articles(articles)
            
            if not clusters:
                logger.error("No article clusters created, aborting")
                return False
            
            logger.info(f"Created {len(clusters)} article clusters")
            
            # 3. Extract topics for each cluster
            logger.info("Extracting topics for clusters")
            topic_clusters = self.content_analyzer.extract_topics(clusters)
            
            # 4. Process articles with LLM (translate to Italian and rewrite)
            logger.info("Processing articles with LLM (translation and rewriting)")
            processed_clusters = self.llm_processor.process_topic_clusters(topic_clusters)
            
            # 5. Generate HTML output and RSS feed
            logger.info("Generating HTML output and RSS feed")
            html_path = self.html_generator.generate_html(processed_clusters)
            
            if not html_path:
                logger.error("HTML generation failed, aborting")
                return False
            
            # 6. Prepare for GitHub Pages publishing
            logger.info("Preparing for GitHub Pages")
            if not self.html_generator.publish_to_github_pages():
                logger.warning("Failed to prepare content for GitHub Pages")
            
            logger.info(f"Newsgator workflow completed successfully. Output generated in {DOCS_DIR}")
            return True
            
        except Exception as e:
            logger.error(f"Error in Newsgator workflow: {str(e)}")
            return False


def main():
    """Main entry point for the Newsgator application."""
    parser = argparse.ArgumentParser(description="Newsgator - RSS feed aggregator with LLM translation")
    parser.add_argument("--version", action="version", version="Newsgator 0.1.0")
    
    args = parser.parse_args()
    
    newsgator = Newsgator()
    success = newsgator.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
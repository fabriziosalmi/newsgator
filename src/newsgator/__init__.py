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

from rich.console import Console
from rich.logging import RichHandler

from newsgator.feed_processing.feed_processor import FeedProcessor
from newsgator.content_analysis.content_analyzer import ContentAnalyzer
from newsgator.llm_integration.llm_processor import LLMProcessor
from newsgator.llm_integration.image_generator import ImageGenerator
from newsgator.html_generation.html_generator import HTMLGenerator
from newsgator.config import DOCS_DIR, ENABLE_IMAGE_GENERATION

# Initialize Rich console
console = Console()

# Setup logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True, console=console, show_time=True, show_path=False),
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
        self.image_generator = ImageGenerator() if ENABLE_IMAGE_GENERATION else None
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
            
            # 5. Generate images for articles (primarily the main article)
            if self.image_generator and self.image_generator.enabled:
                logger.info("Generating images for articles")
                self._generate_images(processed_clusters)
            else:
                logger.info("Image generation is disabled or not available")
            
            # 6. Generate HTML output and RSS feed
            logger.info("Generating HTML output and RSS feed")
            html_path = self.html_generator.generate_html(processed_clusters)
            rss_path = self.html_generator.generate_rss(processed_clusters)
            
            if not html_path:
                logger.error("HTML generation failed, aborting")
                return False
            
            # 7. Log completion summary
            self._log_summary(len(articles), len(clusters), processed_clusters)
            
            logger.info(f"Newsgator workflow completed successfully. Output generated in {DOCS_DIR}")
            return True
            
        except Exception as e:
            logger.error(f"Error in Newsgator workflow: {str(e)}")
            return False
    
    def _generate_images(self, topic_clusters: List[Tuple[List[Dict[str, Any]], str]]) -> None:
        """
        Generate images for articles, focusing on the main article.
        
        Args:
            topic_clusters: List of tuples containing (cluster, topic).
        """
        if not self.image_generator or not self.image_generator.enabled:
            return
            
        try:
            # Find the main article (the one with the most content)
            main_article = None
            main_article_topic = ""
            main_article_size = 0
            
            for cluster, topic in topic_clusters:
                if cluster:
                    article = cluster[0]  # Primary article in each cluster
                    content_length = len(article.get('content', ''))
                    
                    if content_length > main_article_size:
                        main_article = article
                        main_article_topic = topic
                        main_article_size = content_length
            
            # Generate image for the main article
            if main_article:
                logger.info(f"Generating image for main article: '{main_article_topic}'")
                
                # Generate and attach image path to the article
                image_path = self.image_generator.generate_image_for_article(main_article, is_main_article=True)
                
                if image_path:
                    main_article['image_path'] = image_path
                    logger.info(f"Image generated successfully for main article: {image_path}")
                else:
                    logger.warning("Failed to generate image for main article")
            else:
                logger.warning("No main article identified for image generation")
                
        except Exception as e:
            logger.error(f"Error generating images: {str(e)}")
    
    def _log_summary(self, total_articles: int, num_clusters: int, 
                    processed_clusters: List[Tuple[List[Dict[str, Any]], str]]) -> None:
        """
        Log a summary of the Newsgator run.
        
        Args:
            total_articles: Total number of articles fetched.
            num_clusters: Number of clusters created.
            processed_clusters: Processed topic clusters.
        """
        num_processed_articles = sum(len(cluster) for cluster, _ in processed_clusters)
        num_topics = len(processed_clusters)
        
        logger.info("\n" + "="*50)
        logger.info("NEWSGATOR RUN SUMMARY")
        logger.info("="*50)
        logger.info(f"Total articles fetched: {total_articles}")
        logger.info(f"Number of clusters: {num_clusters}")
        logger.info(f"Number of topics: {num_topics}")
        logger.info(f"Number of articles in final output: {num_processed_articles}")
        
        # LLM stats
        if hasattr(self.llm_processor, 'stats'):
            logger.info("\nLLM PROCESSING STATS:")
            for key, value in self.llm_processor.stats.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        # Image generation stats
        if self.image_generator and self.image_generator.enabled:
            logger.info("\nIMAGE GENERATION:")
            logger.info(f"  Model: {self.image_generator.model}")
            
        logger.info("="*50)


def main():
    """Main entry point for the Newsgator application."""
    parser = argparse.ArgumentParser(description="Newsgator - RSS feed aggregator with LLM translation")
    parser.add_argument("--version", action="version", version="Newsgator 0.1.1")
    parser.add_argument("--no-images", action="store_true", help="Disable image generation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Disable image generation if requested
    if args.no_images:
        logger.info("Image generation disabled via command line argument")
        os.environ["ENABLE_IMAGE_GENERATION"] = "false"
    
    # Run the application
    newsgator = Newsgator()
    success = newsgator.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
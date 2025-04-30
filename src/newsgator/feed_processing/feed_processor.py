"""
Feed Processing Module

This module handles fetching and parsing RSS feeds from various news sources.
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
import time
import random

from newsgator.config import RSS_FEEDS

logger = logging.getLogger(__name__)

class FeedProcessor:
    """Class for processing RSS feeds."""
    
    def __init__(self, feed_list: List[Dict[str, str]] = None):
        """
        Initialize the FeedProcessor with a list of feeds.
        
        Args:
            feed_list: List of dictionaries with feed names and URLs.
                       If None, uses the default feeds from config.
        """
        self.feeds = feed_list or RSS_FEEDS
    
    def fetch_all_feeds(self) -> List[Dict[str, Any]]:
        """
        Fetch all configured RSS feeds and extract articles.
        
        Returns:
            List of article dictionaries containing title, description, url, etc.
        """
        all_articles = []
        
        for feed in self.feeds:
            try:
                logger.info(f"Fetching feed: {feed['name']}")
                articles = self.fetch_feed(feed['url'], feed['name'])
                all_articles.extend(articles)
                # Add a small delay to avoid overwhelming servers
                time.sleep(random.uniform(0.5, 1.5))
            except Exception as e:
                logger.error(f"Error fetching feed {feed['name']}: {str(e)}")
        
        return all_articles
    
    def fetch_feed(self, feed_url: str, source_name: str) -> List[Dict[str, Any]]:
        """
        Fetch and parse a single RSS feed.
        
        Args:
            feed_url: URL of the RSS feed to fetch.
            source_name: Name of the feed source.
            
        Returns:
            List of article dictionaries.
        """
        articles = []
        
        try:
            feed_data = feedparser.parse(feed_url)
            
            for entry in feed_data.entries:
                article = self._parse_entry(entry, source_name)
                if article:
                    articles.append(article)
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {str(e)}")
        
        return articles
    
    def _parse_entry(self, entry: Dict[str, Any], source_name: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single entry from a feed into a structured article dictionary.
        
        Args:
            entry: Feed entry from feedparser.
            source_name: Name of the feed source.
            
        Returns:
            Article dictionary or None if parsing fails.
        """
        try:
            # Extract basic fields
            article = {
                'title': entry.get('title', ''),
                'summary': entry.get('summary', ''),
                'url': entry.get('link', ''),
                'source': source_name,
                'published_date': self._parse_date(entry),
                'fetch_date': datetime.now().isoformat(),
                'content': '',
                'categories': [tag.get('term', '') for tag in entry.get('tags', [])] if hasattr(entry, 'tags') else []
            }
            
            # Skip if title or URL is missing
            if not article['title'] or not article['url']:
                return None
            
            # Fetch full article content if available
            full_content = self._fetch_article_content(article['url'])
            if full_content:
                article['content'] = full_content
            
            return article
        except Exception as e:
            logger.error(f"Error parsing entry: {str(e)}")
            return None
    
    def _parse_date(self, entry: Dict[str, Any]) -> str:
        """
        Parse and standardize the publication date.
        
        Args:
            entry: Feed entry from feedparser.
            
        Returns:
            ISO format date string or current date if parsing fails.
        """
        for date_field in ['published', 'updated', 'pubDate']:
            if hasattr(entry, date_field):
                try:
                    if hasattr(entry, f"{date_field}_parsed"):
                        parsed_time = entry.get(f"{date_field}_parsed")
                        if parsed_time:
                            return time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed_time)
                    
                    # Try to parse the date string directly
                    date_str = getattr(entry, date_field)
                    date_obj = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
                    return date_obj.isoformat()
                except (ValueError, AttributeError):
                    pass
        
        # Default to current time if parsing fails
        return datetime.now().isoformat()
    
    def _fetch_article_content(self, url: str) -> str:
        """
        Fetch and extract the full content of an article using BeautifulSoup.
        
        Args:
            url: URL of the article.
            
        Returns:
            Extracted article content or empty string if extraction fails.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts, styles, and other non-content elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav']):
                element.decompose()
            
            # Try to find the main content
            content = ''
            
            # Look for common article containers
            article_element = soup.find('article') or soup.find(class_=lambda c: c and any(x in c.lower() for x in ['article', 'content', 'story', 'entry']))
            
            if article_element:
                # Get paragraphs from the article element
                paragraphs = article_element.find_all('p')
                content = ' '.join(p.get_text().strip() for p in paragraphs)
            else:
                # Fallback to all paragraphs
                paragraphs = soup.find_all('p')
                content = ' '.join(p.get_text().strip() for p in paragraphs)
            
            return content.strip()
        except Exception as e:
            logger.error(f"Error fetching article content from {url}: {str(e)}")
            return ""
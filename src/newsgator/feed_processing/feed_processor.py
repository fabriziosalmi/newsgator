"""
Feed Processing Module

This module handles fetching and parsing RSS feeds from various news sources.
"""

import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import random
import socket
from urllib.parse import urlparse
from http.client import RemoteDisconnected
import ssl

from newsgator.config import (
    RSS_FEEDS, FEED_REQUEST_TIMEOUT, FEED_MAX_RETRIES, 
    FEED_RETRY_DELAY, FEED_MAX_ITEMS_PER_FEED, 
    FEED_FETCH_DELAY_MIN, FEED_FETCH_DELAY_MAX
)

logger = logging.getLogger(__name__)

class FeedProcessor:
    """Class for processing RSS feeds."""
    
    def __init__(self, feed_list: List[Dict[str, Any]] = None):
        """
        Initialize the FeedProcessor with a list of feeds.
        
        Args:
            feed_list: List of dictionaries with feed names and URLs.
                       If None, uses the default feeds from config.
        """
        self.feeds = feed_list or RSS_FEEDS
        self.timeout = FEED_REQUEST_TIMEOUT
        self.max_retries = FEED_MAX_RETRIES
        self.retry_delay = FEED_RETRY_DELAY
        self.max_items_per_feed = FEED_MAX_ITEMS_PER_FEED
        self.fetch_delay_min = FEED_FETCH_DELAY_MIN
        self.fetch_delay_max = FEED_FETCH_DELAY_MAX
        
        # Statistics for reporting
        self.stats = {
            'total_feeds': len(self.feeds),
            'successful_feeds': 0,
            'failed_feeds': 0,
            'skipped_articles': 0,
            'fetched_articles': 0,
            'timeout_errors': 0,
            'connection_errors': 0,
            'parse_errors': 0,
            'other_errors': 0
        }
    
    def fetch_all_feeds(self) -> List[Dict[str, Any]]:
        """
        Fetch all configured RSS feeds and extract articles.
        
        Returns:
            List of article dictionaries containing title, description, url, etc.
        """
        all_articles = []
        
        logger.info(f"Starting to fetch {len(self.feeds)} feeds")
        
        for feed in self.feeds:
            try:
                # Extract feed info
                feed_name = feed.get('name', 'Unknown')
                feed_url = feed.get('url', '')
                feed_category = feed.get('category', 'general')
                
                if not feed_url:
                    logger.warning(f"Skipping feed with missing URL: {feed_name}")
                    continue
                
                logger.info(f"Fetching feed: {feed_name} (Category: {feed_category})")
                articles = self.fetch_feed(feed_url, feed_name, feed_category)
                
                if articles:
                    logger.info(f"Successfully fetched {len(articles)} articles from {feed_name}")
                    all_articles.extend(articles)
                    self.stats['successful_feeds'] += 1
                    self.stats['fetched_articles'] += len(articles)
                else:
                    logger.warning(f"No articles fetched from {feed_name}")
                    self.stats['failed_feeds'] += 1
                
                # Add a random delay to avoid overwhelming servers
                delay = random.uniform(self.fetch_delay_min, self.fetch_delay_max)
                logger.debug(f"Waiting {delay:.2f} seconds before next feed")
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Unexpected error fetching feed {feed.get('name', 'Unknown')}: {str(e)}")
                self.stats['failed_feeds'] += 1
                self.stats['other_errors'] += 1
        
        # Log statistics
        self._log_stats()
        
        return all_articles
    
    def fetch_feed(self, feed_url: str, source_name: str, category: str = "general") -> List[Dict[str, Any]]:
        """
        Fetch and parse a single RSS feed with retry mechanism.
        
        Args:
            feed_url: URL of the RSS feed to fetch.
            source_name: Name of the feed source.
            category: Category of the feed.
            
        Returns:
            List of article dictionaries.
        """
        articles = []
        attempts = 0
        
        while attempts <= self.max_retries:
            try:
                logger.debug(f"Attempting to fetch {feed_url} (Attempt {attempts + 1}/{self.max_retries + 1})")
                
                # Parse feed with timeout
                feed_data = self._parse_feed_with_timeout(feed_url)
                
                if not feed_data or hasattr(feed_data, 'bozo_exception'):
                    bozo_msg = str(feed_data.bozo_exception) if hasattr(feed_data, 'bozo_exception') else "Unknown error"
                    logger.warning(f"Feed parsing error for {source_name}: {bozo_msg}")
                    
                    if attempts < self.max_retries:
                        attempts += 1
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        self.stats['parse_errors'] += 1
                        return articles
                
                logger.debug(f"Feed {source_name} has {len(feed_data.entries)} entries")
                
                # Process each entry up to the maximum limit
                entries_processed = 0
                for entry in feed_data.entries:
                    if entries_processed >= self.max_items_per_feed:
                        logger.debug(f"Reached max items limit ({self.max_items_per_feed}) for {source_name}")
                        self.stats['skipped_articles'] += len(feed_data.entries) - entries_processed
                        break
                    
                    article = self._parse_entry(entry, source_name, category)
                    if article:
                        articles.append(article)
                        entries_processed += 1
                
                return articles
                
            except (socket.timeout, requests.exceptions.Timeout) as e:
                logger.warning(f"Timeout error fetching {source_name}: {str(e)}")
                self.stats['timeout_errors'] += 1
                
            except (requests.exceptions.ConnectionError, RemoteDisconnected, ssl.SSLError) as e:
                logger.warning(f"Connection error fetching {source_name}: {str(e)}")
                self.stats['connection_errors'] += 1
                
            except Exception as e:
                logger.error(f"Error parsing feed {source_name}: {str(e)}")
                self.stats['other_errors'] += 1
            
            # If we got here, we had an error and should retry
            if attempts < self.max_retries:
                attempts += 1
                logger.info(f"Retrying {source_name} in {self.retry_delay} seconds (Attempt {attempts}/{self.max_retries})")
                time.sleep(self.retry_delay)
            else:
                logger.error(f"Failed to fetch {source_name} after {self.max_retries + 1} attempts")
                break
        
        return articles
    
    def _parse_feed_with_timeout(self, feed_url: str) -> Any:
        """
        Parse a feed with a timeout to prevent hanging.
        
        Args:
            feed_url: URL of the RSS feed to fetch.
            
        Returns:
            Parsed feed data or None on failure.
        """
        # Feedparser doesn't directly support timeout, so we need a workaround
        # First, we'll make sure the socket has a timeout
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self.timeout)
        
        try:
            # Check if it's a feed with known encoding issues
            if ("corriereobjects.it" in feed_url or "corriere.it" in feed_url or 
                "ilsole24ore.com" in feed_url or "ilsole24ore" in feed_url):
                logger.debug(f"Using explicit UTF-8 handling for feed with encoding issues: {feed_url}")
                # Fetch the feed content manually with requests and force UTF-8 encoding
                response = requests.get(feed_url, timeout=self.timeout)
                response.encoding = 'utf-8'  # Force UTF-8 encoding
                feed_data = feedparser.parse(response.text)
            else:
                # Regular parsing for other feeds
                feed_data = feedparser.parse(feed_url)
            
            return feed_data
        except Exception as e:
            logger.error(f"Error in feedparser with URL {feed_url}: {str(e)}")
            return None
        finally:
            # Restore the original timeout
            socket.setdefaulttimeout(original_timeout)
    
    def _parse_entry(self, entry: Dict[str, Any], source_name: str, category: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single entry from a feed into a structured article dictionary.
        
        Args:
            entry: Feed entry from feedparser.
            source_name: Name of the feed source.
            category: Category of the feed.
            
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
                'category': category,
                'published_date': self._parse_date(entry),
                'fetch_date': datetime.now().isoformat(),
                'content': '',
                'categories': [tag.get('term', '') for tag in entry.get('tags', [])] if hasattr(entry, 'tags') else []
            }
            
            # Skip if title or URL is missing
            if not article['title'] or not article['url']:
                logger.debug(f"Skipping entry with missing title or URL from {source_name}")
                return None
            
            # Use the entry content if available
            if hasattr(entry, 'content') and entry.content:
                for content in entry.content:
                    if content.get('type') == 'text/html' and content.get('value'):
                        article['content'] = content.get('value')
                        break
            
            # If still no content, try known field names
            if not article['content']:
                for field in ['content', 'description', 'summary']:
                    if hasattr(entry, field) and getattr(entry, field):
                        article['content'] = getattr(entry, field)
                        break
            
            # Fetch full article content if still no content or content is too short
            if not article['content'] or len(article['content']) < 500:
                full_content = self._fetch_article_content(article['url'])
                if full_content:
                    article['content'] = full_content
            
            return article
        except Exception as e:
            logger.error(f"Error parsing entry from {source_name}: {str(e)}")
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
                    if hasattr(entry, f"{date_field}_parsed") and entry.get(f"{date_field}_parsed"):
                        parsed_time = entry.get(f"{date_field}_parsed")
                        return time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed_time)
                    
                    # Try to parse the date string directly
                    date_str = getattr(entry, date_field)
                    # Try different date formats
                    for fmt in [
                        "%a, %d %b %Y %H:%M:%S %z",
                        "%a, %d %b %Y %H:%M:%S %Z",
                        "%Y-%m-%dT%H:%M:%S%z",
                        "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%d %H:%M:%S"
                    ]:
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            return date_obj.isoformat()
                        except ValueError:
                            continue
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
            # Check URL validity
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                logger.warning(f"Invalid URL: {url}")
                return ""
            
            # Use a modern user agent
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': f"{parsed_url.scheme}://{parsed_url.netloc}/",
                'DNT': '1',  # Do Not Track
            }
            
            # Implement retry mechanism
            for attempt in range(self.max_retries + 1):
                try:
                    response = requests.get(
                        url, 
                        headers=headers, 
                        timeout=self.timeout,
                        allow_redirects=True
                    )
                    response.raise_for_status()
                    break  # Successfully got the response
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    if attempt < self.max_retries:
                        logger.debug(f"Retrying article fetch for {url}, attempt {attempt + 1}/{self.max_retries}")
                        time.sleep(self.retry_delay)
                    else:
                        logger.warning(f"Failed to fetch article content after {self.max_retries} retries: {url}")
                        return ""
                except requests.exceptions.HTTPError as e:
                    # Don't retry for 4xx errors (client errors)
                    if 400 <= e.response.status_code < 500:
                        logger.warning(f"HTTP error {e.response.status_code} for {url}: {str(e)}")
                        return ""
                    elif attempt < self.max_retries:
                        logger.debug(f"Retrying after HTTP error {e.response.status_code} for {url}")
                        time.sleep(self.retry_delay)
                    else:
                        logger.warning(f"Failed to fetch article after HTTP errors: {url}")
                        return ""
            
            # Check if we have a valid response
            if not hasattr(response, 'text') or not response.text:
                logger.warning(f"Empty response from {url}")
                return ""
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts, styles, and other non-content elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'iframe', 'form']):
                element.decompose()
            
            # Try different strategies to find the main content
            content = ''
            
            # Strategy 1: Look for common article containers
            article_element = (
                soup.find('article') 
                or soup.find(class_=lambda c: c and any(x in c.lower() for x in ['article', 'content', 'story', 'entry', 'post', 'main']))
                or soup.find(id=lambda i: i and any(x in i.lower() for x in ['article', 'content', 'story', 'entry', 'post', 'main']))
                or soup.find('main')
            )
            
            if article_element:
                # Get paragraphs from the article element
                paragraphs = article_element.find_all('p')
                if paragraphs:
                    content = ' '.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20)
            
            # Strategy 2: If no article element or no paragraphs found, look for all paragraphs
            if not content:
                # Find all paragraphs, but exclude very short ones
                paragraphs = soup.find_all('p')
                if paragraphs:
                    content = ' '.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20)
            
            # Strategy 3: If still no content, get the text from the body
            if not content and soup.body:
                content = soup.body.get_text().strip()
                # Split by newlines and rejoin to get rid of excessive whitespace
                content = ' '.join([line.strip() for line in content.split('\n') if line.strip()])
            
            # Clean up the content
            content = content.strip()
            
            # Check if we have meaningful content
            if len(content) < 100:
                logger.warning(f"Very short content extracted from {url} ({len(content)} chars)")
                
                # If content is too short, also add the title
                if soup.title:
                    title_text = soup.title.get_text().strip()
                    if title_text:
                        content = f"{title_text}\n\n{content}"
            
            return content.strip()
            
        except Exception as e:
            logger.error(f"Error fetching article content from {url}: {str(e)}")
            return ""
    
    def _log_stats(self):
        """Log feed processing statistics."""
        logger.info("Feed processing statistics:")
        logger.info(f"  Total feeds: {self.stats['total_feeds']}")
        logger.info(f"  Successful feeds: {self.stats['successful_feeds']}")
        logger.info(f"  Failed feeds: {self.stats['failed_feeds']}")
        logger.info(f"  Articles fetched: {self.stats['fetched_articles']}")
        logger.info(f"  Articles skipped: {self.stats['skipped_articles']}")
        logger.info(f"  Timeout errors: {self.stats['timeout_errors']}")
        logger.info(f"  Connection errors: {self.stats['connection_errors']}")
        logger.info(f"  Parse errors: {self.stats['parse_errors']}")
        logger.info(f"  Other errors: {self.stats['other_errors']}")
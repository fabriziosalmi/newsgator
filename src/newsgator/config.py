"""
Newsgator Configuration Module

This module contains configuration settings for the Newsgator application.
"""

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
DOCS_DIR = ROOT_DIR / "docs"
TEMPLATES_DIR = Path(__file__).parent / "templates"

# RSS Feed sources
RSS_FEEDS = [
    # News
    {"name": "BBC News", "url": "http://feeds.bbci.co.uk/news/world/rss.xml", "category": "news"},
    {"name": "CNN", "url": "http://rss.cnn.com/rss/edition_world.rss", "category": "news"},
    {"name": "Reuters", "url": "http://feeds.reuters.com/reuters/worldNews", "category": "news"},
    {"name": "The Guardian", "url": "https://www.theguardian.com/world/rss", "category": "news"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml", "category": "news"},
    {"name": "NPR News", "url": "https://feeds.npr.org/1001/rss.xml", "category": "news"},
    {"name": "ABC News", "url": "https://abcnews.go.com/abcnews/internationalheadlines", "category": "news"},
    {"name": "Washington Post", "url": "https://feeds.washingtonpost.com/rss/world", "category": "news"},
    {"name": "Associated Press", "url": "https://feeds.feedburner.com/apnews/topnews", "category": "news"},
    {"name": "Yahoo News", "url": "https://www.yahoo.com/news/rss", "category": "news"},
    
    # Tech
    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/", "category": "tech"},
    {"name": "Wired", "url": "https://www.wired.com/feed/rss", "category": "tech"},
    {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml", "category": "tech"},
    {"name": "Ars Technica", "url": "http://feeds.arstechnica.com/arstechnica/index", "category": "tech"},
    
    # Science
    {"name": "Science Daily", "url": "https://www.sciencedaily.com/rss/all.xml", "category": "science"},
    {"name": "Nature", "url": "https://www.nature.com/nature.rss", "category": "science"},
    {"name": "NASA", "url": "https://www.nasa.gov/rss/dyn/breaking_news.rss", "category": "science"},
    
    # Business & Economics
    {"name": "Financial Times", "url": "https://www.ft.com/?format=rss", "category": "business"},
    {"name": "The Economist", "url": "https://www.economist.com/finance-and-economics/rss.xml", "category": "business"},
    {"name": "Wall Street Journal", "url": "https://feeds.a.dj.com/rss/RSSWorldNews.xml", "category": "business"},
    
]

# Content analysis settings
SIMILARITY_THRESHOLD = 0.7  # Threshold for determining similar articles
MAX_ARTICLES_PER_CATEGORY = 5  # Maximum number of articles to keep per category

# Feed processing settings
FEED_REQUEST_TIMEOUT = 10  # Timeout in seconds for feed requests
FEED_MAX_RETRIES = 2  # Maximum number of retry attempts for failed feed fetches
FEED_RETRY_DELAY = 2  # Delay between retries in seconds
FEED_MAX_ITEMS_PER_FEED = 15  # Maximum number of items to process from each feed
FEED_FETCH_DELAY_MIN = 0.5  # Minimum delay between feed fetches (seconds)
FEED_FETCH_DELAY_MAX = 1.5  # Maximum delay between feed fetches (seconds)

# LLM integration settings
# LLM Provider: "openai" or "lmstudio"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "lmstudio")

# OpenAI settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4"  # or any other OpenAI model you prefer

# LM Studio settings
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "phi-4-mini-instruct")

# Common LLM settings
LLM_TEMPERATURE = 0.5
MAX_TOKENS = 1500

# HTML generation settings
HTML_TITLE = "Newsgator Daily"
HTML_DESCRIPTION = "Daily news aggregated, categorized, and translated."
HTML_AUTHOR = "Newsgator Bot"
HTML_LANGUAGE = "it"  # Target language: Italian

# Output settings
OUTPUT_RSS_FEED = True
RSS_FEED_TITLE = "Newsgator Daily Feed"
RSS_FEED_DESCRIPTION = "Daily news in Italian, aggregated and translated."
RSS_FEED_LINK = "https://example.com/newsgator"  # Will be overridden if served locally

# Web server settings (for Docker)
WEB_SERVER_PORT = 8080
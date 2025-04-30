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
    {"name": "BBC News", "url": "http://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "CNN", "url": "http://rss.cnn.com/rss/edition_world.rss"},
    {"name": "Reuters", "url": "http://feeds.reuters.com/reuters/worldNews"},
    {"name": "The Guardian", "url": "https://www.theguardian.com/world/rss"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml"},
]

# Content analysis settings
SIMILARITY_THRESHOLD = 0.7  # Threshold for determining similar articles
MAX_ARTICLES_PER_CATEGORY = 5  # Maximum number of articles to keep per category

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
"""
Newsgator Configuration Module

This module contains configuration settings for the Newsgator application.
It loads settings from multiple sources in order of priority:
1. Command-line arguments (highest priority)
2. Environment variables
3. Configuration YAML file
4. Default values (lowest priority)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
# Initialize logger
logger = logging.getLogger(__name__)

# Project paths
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
DOCS_DIR = ROOT_DIR / "docs"
TEMPLATES_DIR = Path(__file__).parent / "templates"
CONFIG_FILE_PATH = ROOT_DIR / "config.yaml"
FALLBACK_IMAGE_DIR = ROOT_DIR / "src" / "newsgator" / "assets" / "fallback_images"

# Default configuration values
DEFAULT_CONFIG = {
    # Project paths
    "paths": {
        "docs_dir": str(DOCS_DIR),
        "templates_dir": str(TEMPLATES_DIR),
    },
    
    # RSS Feed sources
    "rss_feeds": [
        {"name": "ANSA", "url": "https://www.ansa.it/sito/notizie/topnews/topnews_rss.xml", "category": "news"},
        {"name": "Corriere della Sera", "url": "https://xml2.corriereobjects.it/rss/homepage.xml", "category": "news"},
        {"name": "La Repubblica", "url": "https://www.repubblica.it/rss/homepage/rss2.0.xml", "category": "news"},
        {"name": "Il Sole 24 Ore", "url": "https://www.ilsole24ore.com/rss/italia--attualita.xml", "category": "news"},
        {"name": "La Stampa", "url": "https://www.lastampa.it/rss.xml", "category": "news"},
    ],
    
    # Content analysis settings
    "content_analysis": {
        "similarity_threshold": 0.51,  # Threshold for determining similar articles
        "max_articles_per_category": 5,  # Maximum number of articles to keep per category
    },
    
    # Feed processing settings
    "feed_processing": {
        "request_timeout": 10,  # Timeout in seconds for feed requests
        "max_retries": 2,  # Maximum number of retry attempts for failed feed fetches
        "retry_delay": 2,  # Delay between retries in seconds
        "max_items_per_feed": 3,  # Maximum number of items to process from each feed
        "fetch_delay_min": 0.5,  # Minimum delay between feed fetches (seconds)
        "fetch_delay_max": 1.5,  # Maximum delay between feed fetches (seconds)
    },
    
    # LLM integration settings
    "llm": {
        "provider": "lmstudio",  # LLM Provider: "openai" or "lmstudio"
        "openai_api_key": "",  # API key for OpenAI
        "openai_model": "gpt-4",  # OpenAI model name
        "lmstudio_base_url": "http://localhost:1234/v1",  # Base URL for LM Studio API
        "lmstudio_model": "phi-4-mini-instruct",  # Model name for LM Studio
        "lmstudio_max_context_length": 32768,  # Max context length for LM Studio
        "temperature": 0.5,  # Temperature setting for generation
        "max_tokens": 1500,  # Maximum tokens for generation
    },
    
    # Image generation settings
    "image_generation": {
        "enabled": True,  # Whether to enable image generation
        "imagen_api_key": "",  # API key for Google Imagen API
        "imagen_model": "imagen-3.0-generate-002",  # Imagen model
        "imagen_sample_count": 1,  # Number of images to generate (limit to main article only)
        "imagen_style": "newspaper black and white photograph",  # Style prefix for prompts
        "imagen_aspect_ratio": "1:1",  # Default aspect ratio (1:1, 3:4, 4:3, 9:16, 16:9)
        "imagen_person_generation": "ALLOW_ADULT",  # ALLOW_ADULT or DONT_ALLOW
        "fallback_image": "placeholder.jpg",  # Default image used if generation fails
    },
    
    # Content length limits
    "content_limits": {
        "max_title_length": 100,  # Maximum length for article titles
        "max_content_length": 3000,  # Maximum length for article content
        "max_main_article_length": 12000,  # Maximum length for main (featured) article content
        "max_summary_length": 500,  # Maximum length for article summaries
        "truncation_suffix": "...",  # Suffix to add when content is truncated
    },
    
    # HTML generation settings
    "html": {
        "title": "Newsgator Daily",  # Title for the HTML output
        "description": "Daily news aggregated, categorized, and translated.",  # Description for the HTML output
        "author": "Newsgator Bot",  # Author for the HTML output
        "language": "it",  # Target language: Italian
        "articles_per_page": 5,  # Number of sections/topics per page
        "multi_page": True,  # Whether to split content into multiple pages
        "page_prefix": "page",  # Prefix for page filenames (page1.html, page2.html, etc.)
        "show_attribution": True,  # Whether to show attribution for sources
        "show_model_info": True,  # Whether to show model information
    },
    
    # RSS feed settings
    "rss": {
        "output_rss_feed": True,  # Whether to output an RSS feed
        "feed_title": "Newsgator Daily Feed",  # Title for the RSS feed
        "feed_description": "Daily news in Italian, aggregated and translated.",  # Description for the RSS feed
        "feed_link": "https://github.com/fabriziosalmi/newsgator",  # Link for the RSS feed
    },
    
    # Web server settings
    "web_server": {
        "port": 8080,  # Port for the web server
    },
}

# Container for the merged configuration
config = {}

def load_config() -> Dict[str, Any]:
    """
    Load configuration from all sources in order of priority:
    1. Command-line arguments (highest priority)
    2. Environment variables
    3. Configuration YAML file
    4. Default values (lowest priority)
    
    Returns:
        Dict[str, Any]: Merged configuration
    """
    global config
    
    # Start with default configuration
    merged_config = DEFAULT_CONFIG.copy()
    
    # Load from YAML file if it exists
    yaml_config = load_yaml_config()
    if yaml_config:
        deep_update(merged_config, yaml_config)
        logger.info(f"Loaded configuration from {CONFIG_FILE_PATH}")
    
    # Load from environment variables
    env_config = load_env_config()
    deep_update(merged_config, env_config)
    
    # Load from command-line arguments
    args_config = load_args_config()
    deep_update(merged_config, args_config)
    
    # Store the merged configuration
    config = merged_config
    
    # Initialize module-level variables from the configuration
    initialize_module_vars(config)
    
    return config

def load_yaml_config() -> Optional[Dict[str, Any]]:
    """
    Load configuration from YAML file.
    
    Returns:
        Optional[Dict[str, Any]]: Configuration from YAML file or None if file doesn't exist
    """
    if not CONFIG_FILE_PATH.exists():
        return None
    
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Error loading configuration from {CONFIG_FILE_PATH}: {str(e)}")
        return None

def load_env_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    Returns:
        Dict[str, Any]: Configuration from environment variables
    """
    env_config = {}
    
    # API keys from environment variables (highest priority)
    # Look for both specific environment variables and general prefix
    api_key_mappings = {
        "IMAGEN_API_KEY": ["image_generation", "imagen_api_key"],
        "NEWSGATOR_IMAGEN_API_KEY": ["image_generation", "imagen_api_key"],
        "OPENAI_API_KEY": ["llm", "openai_api_key"],
        "NEWSGATOR_OPENAI_API_KEY": ["llm", "openai_api_key"],
    }
    
    # Process API keys specifically for higher priority
    for env_var, config_path in api_key_mappings.items():
        if env_var in os.environ and os.environ[env_var]:
            deep_set(env_config, config_path, os.environ[env_var])
    
    # LLM settings
    if "LLM_PROVIDER" in os.environ:
        deep_set(env_config, ["llm", "provider"], os.environ["LLM_PROVIDER"])
    if "OPENAI_MODEL" in os.environ:
        deep_set(env_config, ["llm", "openai_model"], os.environ["OPENAI_MODEL"])
    if "LMSTUDIO_BASE_URL" in os.environ:
        deep_set(env_config, ["llm", "lmstudio_base_url"], os.environ["LMSTUDIO_BASE_URL"])
    if "LMSTUDIO_MODEL" in os.environ:
        deep_set(env_config, ["llm", "lmstudio_model"], os.environ["LMSTUDIO_MODEL"])
    if "LMSTUDIO_MAX_CONTEXT_LENGTH" in os.environ:
        deep_set(env_config, ["llm", "lmstudio_max_context_length"], 
                 int(os.environ["LMSTUDIO_MAX_CONTEXT_LENGTH"]))
    
    # Image generation settings
    if "ENABLE_IMAGE_GENERATION" in os.environ:
        enable_images = os.environ["ENABLE_IMAGE_GENERATION"].lower() == "true"
        deep_set(env_config, ["image_generation", "enabled"], enable_images)
    if "IMAGEN_MODEL" in os.environ:
        deep_set(env_config, ["image_generation", "imagen_model"], os.environ["IMAGEN_MODEL"])
    if "IMAGEN_ASPECT_RATIO" in os.environ:
        deep_set(env_config, ["image_generation", "imagen_aspect_ratio"], os.environ["IMAGEN_ASPECT_RATIO"])
    if "IMAGEN_PERSON_GENERATION" in os.environ:
        deep_set(env_config, ["image_generation", "imagen_person_generation"], 
                 os.environ["IMAGEN_PERSON_GENERATION"])
    if "IMAGEN_FALLBACK_IMAGE" in os.environ:
        deep_set(env_config, ["image_generation", "fallback_image"], 
                 os.environ["IMAGEN_FALLBACK_IMAGE"])
    
    # HTML settings
    if "HTML_LANGUAGE" in os.environ:
        deep_set(env_config, ["html", "language"], os.environ["HTML_LANGUAGE"])
    if "HTML_TITLE" in os.environ:
        deep_set(env_config, ["html", "title"], os.environ["HTML_TITLE"])
    if "HTML_SHOW_ATTRIBUTION" in os.environ:
        show_attr = os.environ["HTML_SHOW_ATTRIBUTION"].lower() == "true"
        deep_set(env_config, ["html", "show_attribution"], show_attr)
    if "HTML_SHOW_MODEL_INFO" in os.environ:
        show_model = os.environ["HTML_SHOW_MODEL_INFO"].lower() == "true"
        deep_set(env_config, ["html", "show_model_info"], show_model)
    
    # Parse other environment variables that match our config structure
    for key, value in os.environ.items():
        if key.startswith("NEWSGATOR_"):
            # Strip the prefix and convert to lowercase
            config_key = key[10:].lower()
            # Split by underscore to get the path
            path = config_key.split("_")
            
            # Try to convert value to appropriate type
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
                value = float(value)
                
            deep_set(env_config, path, value)
    
    return env_config

def load_args_config() -> Dict[str, Any]:
    """
    Load configuration from command-line arguments.
    
    Returns:
        Dict[str, Any]: Configuration from command-line arguments
    """
    args_config = {}
    
    # Only parse args if we're running as the main module
    if not sys.argv[0].endswith("config.py"):
        try:
            # This is a simple parser to avoid conflicts with the main parser
            parser = argparse.ArgumentParser(add_help=False)
            
            # API key options
            parser.add_argument("--imagen-api-key", help="Google Imagen API key")
            parser.add_argument("--openai-api-key", help="OpenAI API key")
            
            # LLM settings
            parser.add_argument("--llm-provider", help="LLM provider (openai or lmstudio)")
            parser.add_argument("--openai-model", help="OpenAI model")
            parser.add_argument("--lmstudio-base-url", help="LM Studio base URL")
            parser.add_argument("--lmstudio-model", help="LM Studio model")
            
            # Image generation settings
            parser.add_argument("--enable-image-generation", dest="enable_image_generation", 
                                action="store_true", help="Enable image generation")
            parser.add_argument("--disable-image-generation", dest="enable_image_generation", 
                                action="store_false", help="Disable image generation")
            parser.add_argument("--imagen-model", help="Google Imagen model")
            parser.add_argument("--imagen-aspect-ratio", help="Imagen aspect ratio")
            parser.add_argument("--imagen-fallback-image", help="Fallback image to use when generation fails")
            
            # HTML settings
            parser.add_argument("--html-language", help="Target language for HTML output")
            parser.add_argument("--html-title", help="Title for HTML output")
            parser.add_argument("--articles-per-page", type=int, help="Number of articles per page")
            parser.add_argument("--single-page", dest="multi_page", action="store_false", 
                                help="Output as a single page")
            parser.add_argument("--show-attribution", dest="show_attribution", action="store_true",
                                help="Show source attribution in articles")
            parser.add_argument("--hide-attribution", dest="show_attribution", action="store_false",
                                help="Hide source attribution in articles")
            parser.add_argument("--show-model-info", dest="show_model_info", action="store_true",
                                help="Show LLM model information in articles")
            parser.add_argument("--hide-model-info", dest="show_model_info", action="store_false",
                                help="Hide LLM model information in articles")
            
            # Content analysis settings
            parser.add_argument("--similarity-threshold", type=float, 
                                help="Threshold for determining similar articles")
            parser.add_argument("--max-articles-per-category", type=int, 
                                help="Maximum articles to keep per category")
            
            # Config file path
            parser.add_argument("--config", help="Path to configuration YAML file")
            
            # Parse known args only, ignoring unknown ones
            args, _ = parser.parse_known_args()
            
            # Convert args to dictionary
            args_dict = vars(args)
            
            # Handle custom config file path
            if args_dict.get("config"):
                custom_config_path = Path(args_dict["config"])
                if custom_config_path.exists():
                    try:
                        with open(custom_config_path, 'r') as f:
                            custom_config = yaml.safe_load(f)
                            deep_update(args_config, custom_config)
                            logger.info(f"Loaded configuration from {custom_config_path}")
                    except Exception as e:
                        logger.warning(f"Error loading custom configuration from {custom_config_path}: {str(e)}")
            
            # Remove None values
            args_dict = {k: v for k, v in args_dict.items() if v is not None and k != "config"}
            
            # Map args to config structure
            for key, value in args_dict.items():
                if key == "imagen_api_key":
                    deep_set(args_config, ["image_generation", "imagen_api_key"], value)
                elif key == "openai_api_key":
                    deep_set(args_config, ["llm", "openai_api_key"], value)
                elif key == "llm_provider":
                    deep_set(args_config, ["llm", "provider"], value)
                elif key == "openai_model":
                    deep_set(args_config, ["llm", "openai_model"], value)
                elif key == "lmstudio_base_url":
                    deep_set(args_config, ["llm", "lmstudio_base_url"], value)
                elif key == "lmstudio_model":
                    deep_set(args_config, ["llm", "lmstudio_model"], value)
                elif key == "enable_image_generation":
                    deep_set(args_config, ["image_generation", "enabled"], value)
                elif key == "imagen_model":
                    deep_set(args_config, ["image_generation", "imagen_model"], value)
                elif key == "imagen_aspect_ratio":
                    deep_set(args_config, ["image_generation", "imagen_aspect_ratio"], value)
                elif key == "imagen_fallback_image":
                    deep_set(args_config, ["image_generation", "fallback_image"], value)
                elif key == "html_language":
                    deep_set(args_config, ["html", "language"], value)
                elif key == "html_title":
                    deep_set(args_config, ["html", "title"], value)
                elif key == "articles_per_page":
                    deep_set(args_config, ["html", "articles_per_page"], value)
                elif key == "multi_page":
                    deep_set(args_config, ["html", "multi_page"], value)
                elif key == "show_attribution":
                    deep_set(args_config, ["html", "show_attribution"], value)
                elif key == "show_model_info":
                    deep_set(args_config, ["html", "show_model_info"], value)
                elif key == "similarity_threshold":
                    deep_set(args_config, ["content_analysis", "similarity_threshold"], value)
                elif key == "max_articles_per_category":
                    deep_set(args_config, ["content_analysis", "max_articles_per_category"], value)
                
        except Exception as e:
            logger.warning(f"Error parsing command-line arguments: {str(e)}")
    
    return args_config

def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a dictionary with another dictionary.
    
    Args:
        d: Dictionary to update
        u: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    if not isinstance(u, dict):
        return u
    
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = deep_update(d[k], v)
        else:
            d[k] = v
    return d

def deep_set(d: Dict[str, Any], path: list, value: Any) -> None:
    """
    Set a value in a nested dictionary by path.
    
    Args:
        d: Dictionary to update
        path: Path to the value as a list of keys
        value: Value to set
    """
    current = d
    for part in path[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]
    current[path[-1]] = value

def initialize_module_vars(config: Dict[str, Any]) -> None:
    """
    Initialize module-level variables from the configuration.
    
    Args:
        config: Configuration dictionary
    """
    # These are the module-level variables that will be used by other modules
    # Paths
    global ROOT_DIR, SRC_DIR, DOCS_DIR, TEMPLATES_DIR
    DOCS_DIR = Path(config["paths"]["docs_dir"])
    TEMPLATES_DIR = Path(config["paths"]["templates_dir"])
    
    # RSS Feed sources
    global RSS_FEEDS
    RSS_FEEDS = config["rss_feeds"]
    
    # Content analysis settings
    global SIMILARITY_THRESHOLD, MAX_ARTICLES_PER_CATEGORY
    SIMILARITY_THRESHOLD = config["content_analysis"]["similarity_threshold"]
    MAX_ARTICLES_PER_CATEGORY = config["content_analysis"]["max_articles_per_category"]
    
    # Feed processing settings
    global FEED_REQUEST_TIMEOUT, FEED_MAX_RETRIES, FEED_RETRY_DELAY
    global FEED_MAX_ITEMS_PER_FEED, FEED_FETCH_DELAY_MIN, FEED_FETCH_DELAY_MAX
    FEED_REQUEST_TIMEOUT = config["feed_processing"]["request_timeout"]
    FEED_MAX_RETRIES = config["feed_processing"]["max_retries"]
    FEED_RETRY_DELAY = config["feed_processing"]["retry_delay"]
    FEED_MAX_ITEMS_PER_FEED = config["feed_processing"]["max_items_per_feed"]
    FEED_FETCH_DELAY_MIN = config["feed_processing"]["fetch_delay_min"]
    FEED_FETCH_DELAY_MAX = config["feed_processing"]["fetch_delay_max"]
    
    # LLM integration settings
    global LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL
    global LMSTUDIO_BASE_URL, LMSTUDIO_MODEL, LMSTUDIO_MAX_CONTEXT_LENGTH
    global LLM_TEMPERATURE, MAX_TOKENS
    LLM_PROVIDER = config["llm"]["provider"]
    OPENAI_API_KEY = config["llm"]["openai_api_key"]
    OPENAI_MODEL = config["llm"]["openai_model"]
    LMSTUDIO_BASE_URL = config["llm"]["lmstudio_base_url"]
    LMSTUDIO_MODEL = config["llm"]["lmstudio_model"]
    LMSTUDIO_MAX_CONTEXT_LENGTH = config["llm"]["lmstudio_max_context_length"]
    LLM_TEMPERATURE = config["llm"]["temperature"]
    MAX_TOKENS = config["llm"]["max_tokens"]
    
    # Image generation settings
    global ENABLE_IMAGE_GENERATION, IMAGEN_API_KEY, IMAGEN_MODEL
    global IMAGEN_SAMPLE_COUNT, IMAGEN_STYLE, IMAGEN_ASPECT_RATIO, IMAGEN_PERSON_GENERATION
    global IMAGEN_FALLBACK_IMAGE
    ENABLE_IMAGE_GENERATION = config["image_generation"]["enabled"]
    IMAGEN_API_KEY = config["image_generation"]["imagen_api_key"]
    IMAGEN_MODEL = config["image_generation"]["imagen_model"]
    IMAGEN_SAMPLE_COUNT = config["image_generation"]["imagen_sample_count"]
    IMAGEN_STYLE = config["image_generation"]["imagen_style"]
    IMAGEN_ASPECT_RATIO = config["image_generation"]["imagen_aspect_ratio"]
    IMAGEN_PERSON_GENERATION = config["image_generation"]["imagen_person_generation"]
    IMAGEN_FALLBACK_IMAGE = config["image_generation"]["fallback_image"]
    
    # Content length limits
    global MAX_TITLE_LENGTH, MAX_CONTENT_LENGTH, MAX_MAIN_ARTICLE_LENGTH
    global MAX_SUMMARY_LENGTH, TRUNCATION_SUFFIX
    MAX_TITLE_LENGTH = config["content_limits"]["max_title_length"]
    MAX_CONTENT_LENGTH = config["content_limits"]["max_content_length"]
    MAX_MAIN_ARTICLE_LENGTH = config["content_limits"]["max_main_article_length"]
    MAX_SUMMARY_LENGTH = config["content_limits"]["max_summary_length"]
    TRUNCATION_SUFFIX = config["content_limits"]["truncation_suffix"]
    
    # HTML generation settings
    global HTML_TITLE, HTML_DESCRIPTION, HTML_AUTHOR, HTML_LANGUAGE
    global HTML_ARTICLES_PER_PAGE, HTML_MULTI_PAGE, HTML_PAGE_PREFIX
    global HTML_SHOW_ATTRIBUTION, HTML_SHOW_MODEL_INFO
    HTML_TITLE = config["html"]["title"]
    HTML_DESCRIPTION = config["html"]["description"]
    HTML_AUTHOR = config["html"]["author"]
    HTML_LANGUAGE = config["html"]["language"]
    HTML_ARTICLES_PER_PAGE = config["html"]["articles_per_page"]
    HTML_MULTI_PAGE = config["html"]["multi_page"]
    HTML_PAGE_PREFIX = config["html"]["page_prefix"]
    HTML_SHOW_ATTRIBUTION = config["html"]["show_attribution"]
    HTML_SHOW_MODEL_INFO = config["html"]["show_model_info"]
    
    # RSS feed settings
    global OUTPUT_RSS_FEED, RSS_FEED_TITLE, RSS_FEED_DESCRIPTION, RSS_FEED_LINK
    OUTPUT_RSS_FEED = config["rss"]["output_rss_feed"]
    RSS_FEED_TITLE = config["rss"]["feed_title"]
    RSS_FEED_DESCRIPTION = config["rss"]["feed_description"]
    RSS_FEED_LINK = config["rss"]["feed_link"]
    
    # Web server settings
    global WEB_SERVER_PORT
    WEB_SERVER_PORT = config["web_server"]["port"]

def create_default_config_file() -> None:
    """
    Create a default configuration file if it doesn't exist.
    """
    if CONFIG_FILE_PATH.exists():
        return
    
    try:
        with open(CONFIG_FILE_PATH, 'w') as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Created default configuration file: {CONFIG_FILE_PATH}")
    except Exception as e:
        logger.warning(f"Error creating default configuration file: {str(e)}")

def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    
    Returns:
        Dict[str, Any]: Current configuration
    """
    global config
    if not config:
        load_config()
    return config

# Load configuration when the module is imported
load_config()

# Create default configuration file if it doesn't exist
create_default_config_file()

# The following variables are initialized by load_config() based on the merged configuration

# RSS Feed sources
RSS_FEEDS = get_config()["rss_feeds"]

# Content analysis settings
SIMILARITY_THRESHOLD = get_config()["content_analysis"]["similarity_threshold"]
MAX_ARTICLES_PER_CATEGORY = get_config()["content_analysis"]["max_articles_per_category"]

# Feed processing settings
FEED_REQUEST_TIMEOUT = get_config()["feed_processing"]["request_timeout"]
FEED_MAX_RETRIES = get_config()["feed_processing"]["max_retries"]
FEED_RETRY_DELAY = get_config()["feed_processing"]["retry_delay"]
FEED_MAX_ITEMS_PER_FEED = get_config()["feed_processing"]["max_items_per_feed"]
FEED_FETCH_DELAY_MIN = get_config()["feed_processing"]["fetch_delay_min"]
FEED_FETCH_DELAY_MAX = get_config()["feed_processing"]["fetch_delay_max"]

# LLM integration settings
LLM_PROVIDER = get_config()["llm"]["provider"]
OPENAI_API_KEY = get_config()["llm"]["openai_api_key"]
OPENAI_MODEL = get_config()["llm"]["openai_model"]
LMSTUDIO_BASE_URL = get_config()["llm"]["lmstudio_base_url"]
LMSTUDIO_MODEL = get_config()["llm"]["lmstudio_model"]
LMSTUDIO_MAX_CONTEXT_LENGTH = get_config()["llm"]["lmstudio_max_context_length"]
LLM_TEMPERATURE = get_config()["llm"]["temperature"]
MAX_TOKENS = get_config()["llm"]["max_tokens"]

# Image generation settings
ENABLE_IMAGE_GENERATION = get_config()["image_generation"]["enabled"]
IMAGEN_API_KEY = get_config()["image_generation"]["imagen_api_key"]
IMAGEN_MODEL = get_config()["image_generation"]["imagen_model"]
IMAGEN_SAMPLE_COUNT = get_config()["image_generation"]["imagen_sample_count"]
IMAGEN_STYLE = get_config()["image_generation"]["imagen_style"]
IMAGEN_ASPECT_RATIO = get_config()["image_generation"]["imagen_aspect_ratio"]
IMAGEN_PERSON_GENERATION = get_config()["image_generation"]["imagen_person_generation"]
IMAGEN_FALLBACK_IMAGE = get_config()["image_generation"]["fallback_image"]

# Content length limits
MAX_TITLE_LENGTH = get_config()["content_limits"]["max_title_length"]
MAX_CONTENT_LENGTH = get_config()["content_limits"]["max_content_length"]
MAX_MAIN_ARTICLE_LENGTH = get_config()["content_limits"]["max_main_article_length"]
MAX_SUMMARY_LENGTH = get_config()["content_limits"]["max_summary_length"]
TRUNCATION_SUFFIX = get_config()["content_limits"]["truncation_suffix"]

# HTML generation settings
HTML_TITLE = get_config()["html"]["title"]
HTML_DESCRIPTION = get_config()["html"]["description"]
HTML_AUTHOR = get_config()["html"]["author"]
HTML_LANGUAGE = get_config()["html"]["language"]
HTML_ARTICLES_PER_PAGE = get_config()["html"]["articles_per_page"]
HTML_MULTI_PAGE = get_config()["html"]["multi_page"]
HTML_PAGE_PREFIX = get_config()["html"]["page_prefix"]
HTML_SHOW_ATTRIBUTION = get_config()["html"]["show_attribution"]
HTML_SHOW_MODEL_INFO = get_config()["html"]["show_model_info"]

# RSS feed settings
OUTPUT_RSS_FEED = get_config()["rss"]["output_rss_feed"]
RSS_FEED_TITLE = get_config()["rss"]["feed_title"]
RSS_FEED_DESCRIPTION = get_config()["rss"]["feed_description"]
RSS_FEED_LINK = get_config()["rss"]["feed_link"]

# Web server settings
WEB_SERVER_PORT = get_config()["web_server"]["port"]
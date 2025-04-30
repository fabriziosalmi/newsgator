"""
Image Generator Module

This module generates images for articles using Google's Imagen API (Gemini).
"""

import os
import logging
import base64
import uuid
import time
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import requests
import hashlib

from newsgator.config import (
    ENABLE_IMAGE_GENERATION, DOCS_DIR, IMAGEN_API_KEY,
    IMAGEN_MODEL, IMAGEN_SAMPLE_COUNT, IMAGEN_STYLE,
    IMAGEN_ASPECT_RATIO, IMAGEN_PERSON_GENERATION,
    FALLBACK_IMAGE_DIR
)

# Initialize logger
logger = logging.getLogger(__name__)

class ImageGenerator:
    """Class for generating images for articles using Google's Imagen API."""
    
    def __init__(self, use_cache=True):
        """
        Initialize the image generator with configuration options.
        
        Args:
            use_cache: Whether to use cached images to avoid regenerating the same image.
        """
        self.enabled = ENABLE_IMAGE_GENERATION
        self.use_cache = use_cache
        
        # Check if API key is available
        if not IMAGEN_API_KEY and self.enabled:
            logger.warning("Imagen API key not found. Image generation will be disabled.")
            self.enabled = False
        
        # Set up directories
        self.image_dir = DOCS_DIR / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up cache directory
        self.cache_dir = DOCS_DIR / ".cache" / "images"
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Image cache directory: {self.cache_dir}")
        
        # Set up fallback images directory
        self.fallback_image_dir = Path(FALLBACK_IMAGE_DIR) if FALLBACK_IMAGE_DIR else None
        if self.fallback_image_dir and self.fallback_image_dir.exists():
            logger.info(f"Fallback image directory: {self.fallback_image_dir}")
            # Check if we have any fallback images
            self.fallback_images = list(self.fallback_image_dir.glob("*.jpg")) + list(self.fallback_image_dir.glob("*.png"))
            if not self.fallback_images:
                logger.warning(f"No fallback images found in {self.fallback_image_dir}")
        else:
            self.fallback_images = []
            if self.enabled:
                logger.warning("No fallback image directory configured or directory does not exist")
        
        # Set up API parameters
        self.api_key = IMAGEN_API_KEY
        self.model = IMAGEN_MODEL
        self.sample_count = IMAGEN_SAMPLE_COUNT
        self.style_prefix = IMAGEN_STYLE
        self.aspect_ratio = IMAGEN_ASPECT_RATIO
        self.person_mode = IMAGEN_PERSON_GENERATION
        
        # API URL for Google Imagen
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        # Retry parameters
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.max_backoff = 30  # maximum backoff in seconds
        
        # Keep track of generation attempts
        self.generation_attempts = 0
        self.generation_successes = 0
        self.generation_failures = 0
        self.cache_hits = 0
        self.fallback_uses = 0
        self.prompt_tokens = 0
        
        if self.enabled:
            logger.info(f"Image generator initialized with model: {self.model}")
            logger.info(f"Images will be saved to: {self.image_dir}")
            logger.info(f"Image caching is {'enabled' if self.use_cache else 'disabled'}")
            logger.info(f"Found {len(self.fallback_images)} fallback images")
        else:
            logger.info("Image generation is disabled")
    
    def generate_image_for_article(self, article: Dict[str, Any], is_main_article: bool = False) -> Optional[str]:
        """
        Generate an image for an article and return the path to the image.
        
        Args:
            article: The article for which to generate an image.
            is_main_article: Whether this is the main article.
            
        Returns:
            Path to the generated image, or None if generation failed.
        """
        if not self.enabled:
            return None
        
        # First, calculate a unique hash for the article content
        article_hash = self._calculate_article_hash(article)
        
        # Check if we have a cached image for this article
        if self.use_cache:
            cached_image = self._get_cached_image(article_hash, is_main_article)
            if cached_image:
                self.cache_hits += 1
                logger.info(f"Using cached image for article: {article.get('title', 'Unknown')}")
                return cached_image
        
        # Generate a prompt based on the article content
        prompt = self._generate_prompt(article, is_main_article)
        
        if not prompt:
            logger.warning("Could not generate prompt for article")
            return self._use_fallback_image(is_main_article)
        
        # Generate a unique filename
        filename = f"article_{article_hash[:8]}_{uuid.uuid4().hex[:4]}.jpg"
        output_path = self.image_dir / filename
        relative_path = f"images/{filename}"
        
        try:
            # Generate image using the Google Imagen API
            image_data = self._call_imagen_api(prompt)
            
            if image_data:
                # Save the image
                with open(output_path, "wb") as f:
                    f.write(image_data)
                
                logger.info(f"Image saved to: {output_path}")
                
                # Cache the image
                if self.use_cache:
                    self._cache_image(article_hash, is_main_article, relative_path)
                
                # Return the relative path from docs directory
                return relative_path
            else:
                logger.warning(f"Failed to generate image for article: {article.get('title', 'Unknown')}")
                self.generation_failures += 1
                return self._use_fallback_image(is_main_article)
                
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            self.generation_failures += 1
            return self._use_fallback_image(is_main_article)
    
    def _calculate_article_hash(self, article: Dict[str, Any]) -> str:
        """
        Calculate a hash for an article to use for caching.
        
        Args:
            article: The article to hash.
            
        Returns:
            A hash string representing the article content.
        """
        title = article.get('title', '')
        content_preview = ' '.join((article.get('content', '')).split()[:100])
        
        # Create a hash of the title and content preview
        content_to_hash = f"{title}|{content_preview}"
        return hashlib.md5(content_to_hash.encode('utf-8')).hexdigest()
    
    def _get_cached_image(self, article_hash: str, is_main_article: bool) -> Optional[str]:
        """
        Check if there's a cached image for this article.
        
        Args:
            article_hash: Hash of the article content.
            is_main_article: Whether this is a main article.
            
        Returns:
            Path to the cached image or None if not found.
        """
        if not self.use_cache:
            return None
            
        cache_key = f"{article_hash}_{'main' if is_main_article else 'regular'}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    image_path = cache_data.get('image_path')
                    
                    # Verify the image still exists
                    full_path = DOCS_DIR / image_path
                    if full_path.exists():
                        return image_path
                    else:
                        logger.warning(f"Cached image file not found: {full_path}")
                        return None
            except Exception as e:
                logger.warning(f"Error reading image cache: {str(e)}")
                
        return None
    
    def _cache_image(self, article_hash: str, is_main_article: bool, image_path: str) -> None:
        """
        Cache an image path for this article.
        
        Args:
            article_hash: Hash of the article content.
            is_main_article: Whether this is a main article.
            image_path: Path to the generated image.
        """
        if not self.use_cache:
            return
            
        cache_key = f"{article_hash}_{'main' if is_main_article else 'regular'}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                'article_hash': article_hash,
                'is_main_article': is_main_article,
                'image_path': image_path,
                'timestamp': time.time()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
                
            logger.debug(f"Cached image path: {image_path}")
        except Exception as e:
            logger.warning(f"Error caching image: {str(e)}")
    
    def _use_fallback_image(self, is_main_article: bool) -> Optional[str]:
        """
        Use a fallback image when generation fails.
        
        Args:
            is_main_article: Whether this is the main article.
            
        Returns:
            Path to a fallback image or None if no fallbacks available.
        """
        if not self.fallback_images:
            return None
            
        # Choose a random fallback image
        fallback_image = random.choice(self.fallback_images)
        
        # Copy the fallback image to the image directory with a unique name
        filename = f"fallback_{uuid.uuid4().hex[:8]}.jpg"
        output_path = self.image_dir / filename
        
        try:
            # Read the fallback image
            with open(fallback_image, "rb") as src_file:
                image_data = src_file.read()
                
            # Write to the output path
            with open(output_path, "wb") as dst_file:
                dst_file.write(image_data)
                
            self.fallback_uses += 1
            logger.info(f"Using fallback image: {fallback_image.name} -> {filename}")
            
            # Return the relative path
            return f"images/{filename}"
        except Exception as e:
            logger.error(f"Error using fallback image: {str(e)}")
            return None
    
    def _generate_prompt(self, article: Dict[str, Any], is_main_article: bool) -> Optional[str]:
        """
        Generate a prompt for image generation based on article content.
        
        Args:
            article: Article for which to generate a prompt.
            is_main_article: Whether this is the main article.
            
        Returns:
            Generated prompt, or None if content is insufficient.
        """
        title = article.get('title', '')
        content = article.get('content', '')
        
        if not title and not content:
            logger.warning("Cannot generate prompt: article has no title or content")
            return None
        
        # Use title and first 50-100 words of content for prompt
        content_preview = ' '.join(content.split()[:100]) if content else ''
        
        prompt = f"{self.style_prefix}, "
        
        if is_main_article:
            # For main article, create a more specific and detailed prompt
            prompt += f"professional editorial photograph for newspaper headline story about '{title}'. "
            
            # Add content-based context if available
            if content_preview:
                prompt += f"The photo should show the following scenario: {content_preview}"
            else:
                prompt += f"A dramatic or impactful scene related to {title}."
        else:
            # For regular articles, create a simpler prompt
            prompt += f"newspaper photograph for article about '{title}'. "
            
            if content_preview:
                # Extract key phrases or sentences for context
                key_phrases = self._extract_key_phrases(content_preview)
                prompt += f"Context: {key_phrases}"
            else:
                prompt += f"Show a relevant scene for this topic."
        
        # Add formatting directives
        prompt += " Ensure the image is newspaper-style with good contrast and lighting."
        
        # Add person mode directive if configured
        if self.person_mode and self.person_mode.lower() != "none":
            prompt += f" {self.person_mode}."
        
        # Estimate token count (rough estimation)
        self.prompt_tokens += len(prompt.split())
        
        logger.debug(f"Generated image prompt: {prompt}")
        return prompt
    
    def _extract_key_phrases(self, text: str) -> str:
        """
        Extract key phrases from text to improve prompt quality.
        
        Args:
            text: Text to extract phrases from.
            
        Returns:
            String with key phrases.
        """
        # Simple extraction of first sentence and any phrases with numbers or named entities
        sentences = text.split('.')
        
        if not sentences:
            return text
            
        key_info = [sentences[0]]  # Always include first sentence
        
        # Look for phrases with numbers, dates, or proper nouns (simple heuristic)
        for sentence in sentences[1:3]:  # Check a few more sentences
            words = sentence.strip().split()
            has_number = any(w.isdigit() or w.replace('.', '').isdigit() for w in words)
            has_proper_noun = any(w[0].isupper() and w.lower() not in ['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for'] for w in words if w)
            
            if has_number or has_proper_noun:
                key_info.append(sentence)
        
        return '. '.join(key_info)
    
    def _call_imagen_api(self, prompt: str) -> Optional[bytes]:
        """
        Call the Google Imagen API to generate an image.
        
        Args:
            prompt: The prompt for image generation.
            
        Returns:
            Image data as bytes, or None if generation failed.
        """
        self.generation_attempts += 1
        
        if not self.api_key:
            logger.error("No API key available for Imagen")
            return None
        
        # Create the request payload
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "sampleCount": self.sample_count
            }
        }
        
        # Add image generation parameters
        if self.aspect_ratio:
            payload["generationConfig"]["aspectRatio"] = self.aspect_ratio
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Retry mechanism for API calls with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Add timeout to prevent hanging requests
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract the image data
                    try:
                        # Navigate through the response to find the image data
                        if 'candidates' in data and data['candidates']:
                            for candidate in data['candidates']:
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    for part in candidate['content']['parts']:
                                        if 'inlineData' in part and 'data' in part['inlineData']:
                                            # Extract and decode the base64 data
                                            image_data = base64.b64decode(part['inlineData']['data'])
                                            self.generation_successes += 1
                                            return image_data
                        
                        logger.warning(f"No image data found in response: {data}")
                        return None
                        
                    except Exception as e:
                        logger.error(f"Error parsing image response: {str(e)}")
                        return None
                
                elif response.status_code == 429:  # Rate limit
                    # Use exponential backoff
                    wait_time = min(self.max_backoff, (2 ** attempt) * self.retry_delay)
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry.")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 503:  # Service unavailable
                    wait_time = min(self.max_backoff, (2 ** attempt) * self.retry_delay)
                    logger.warning(f"Service temporarily unavailable. Waiting {wait_time} seconds before retry.")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 504:  # Gateway timeout
                    wait_time = min(self.max_backoff, (2 ** attempt) * self.retry_delay * 2)  # Wait longer for timeouts
                    logger.warning(f"Gateway timeout. Waiting {wait_time} seconds before retry.")
                    time.sleep(wait_time)
                    continue   
                else:
                    error_message = f"API error: {response.status_code}"
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            error_message += f" - {error_data['error'].get('message', '')}"
                    except:
                        error_message += f" - {response.text[:100]}"
                    
                    logger.error(error_message)
                    
                    # For some errors, retrying won't help
                    if response.status_code in [400, 401, 403]:
                        return None
                    
                    # For other errors, try again with backoff
                    if attempt < self.max_retries - 1:
                        wait_time = min(self.max_backoff, (2 ** attempt) * self.retry_delay)
                        logger.warning(f"API error. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        return None
            except requests.exceptions.Timeout:
                logger.error("API request timed out")
                if attempt < self.max_retries - 1:
                    wait_time = min(self.max_backoff, (2 ** attempt) * self.retry_delay * 2)
                    logger.warning(f"Timeout occurred. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None
            except requests.exceptions.ConnectionError:
                logger.error("Connection error when calling API")
                if attempt < self.max_retries - 1:
                    wait_time = min(self.max_backoff, (2 ** attempt) * self.retry_delay)
                    logger.warning(f"Connection error. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None            
            except Exception as e:
                logger.error(f"Error calling Imagen API: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = min(self.max_backoff, (2 ** attempt) * self.retry_delay)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None
        
        # If we've exhausted all retries
        return None
    
    def batch_generate_images(self, articles: List[Dict[str, Any]], main_article_index: int = 0) -> Dict[str, str]:
        """
        Generate images for multiple articles in batch.
        
        Args:
            articles: List of articles to generate images for.
            main_article_index: Index of the main article in the list.
            
        Returns:
            Dictionary mapping article IDs to image paths.
        """
        if not self.enabled or not articles:
            return {}
            
        result = {}
        total_articles = len(articles)
        
        logger.info(f"Batch generating images for {total_articles} articles")
        
        for i, article in enumerate(articles):
            article_id = article.get('id', str(i))
            is_main = (i == main_article_index)
            
            logger.info(f"Generating image for article {i+1}/{total_articles}: {article.get('title', 'Unknown')}")
            
            # Generate the image
            image_path = self.generate_image_for_article(article, is_main)
            
            if image_path:
                result[article_id] = image_path
                
            # Add a small delay between requests to avoid rate limiting
            if i < total_articles - 1:
                time.sleep(0.5)
                
        logger.info(f"Batch image generation complete. Generated {len(result)}/{total_articles} images")
        return result
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about image generation.
        
        Returns:
            Dictionary with statistics.
        """
        success_rate = (self.generation_successes / self.generation_attempts * 100) if self.generation_attempts else 0
        
        return {
            "attempts": self.generation_attempts,
            "successes": self.generation_successes,
            "failures": self.generation_failures,
            "cache_hits": self.cache_hits,
            "fallback_uses": self.fallback_uses,
            "success_rate": success_rate,
            "prompt_tokens": self.prompt_tokens,
            "model": self.model,
            "enabled": self.enabled,
            "cache_enabled": self.use_cache
        }
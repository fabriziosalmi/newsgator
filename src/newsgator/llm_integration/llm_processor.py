"""
LLM Integration Module

This module handles integration with LLMs for content translation and rewriting.
Supports both OpenAI and local LM Studio models.
"""

import logging
import os
import json
import requests
import re
import math
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import time
from pathlib import Path

from newsgator.config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, 
    LMSTUDIO_BASE_URL, LMSTUDIO_MODEL, LMSTUDIO_MAX_CONTEXT_LENGTH,
    LLM_TEMPERATURE, MAX_TOKENS, HTML_LANGUAGE,
    MAX_TITLE_LENGTH, MAX_CONTENT_LENGTH, MAX_SUMMARY_LENGTH, TRUNCATION_SUFFIX,
    DOCS_DIR
)

logger = logging.getLogger(__name__)

class LLMProcessor:
    """Class for processing content with LLMs."""
    
    def __init__(
        self, 
        provider=None,
        api_key=None, 
        model=None, 
        base_url=None,
        temperature=None, 
        max_tokens=None, 
        target_language=None,
        enable_caching=True
    ):
        """
        Initialize the LLM processor.
        
        Args:
            provider: LLM provider to use ("openai" or "lmstudio").
            api_key: API key for OpenAI (not needed for LM Studio).
            model: The LLM model to use.
            base_url: Base URL for LM Studio API.
            temperature: Temperature setting for generation.
            max_tokens: Maximum tokens for generation.
            target_language: Target language for translation.
            enable_caching: Whether to enable caching of LLM responses.
        """
        self.provider = provider or LLM_PROVIDER
        self.api_key = api_key or OPENAI_API_KEY
        self.temperature = temperature or LLM_TEMPERATURE
        self.enable_caching = enable_caching
        
        # Set default max tokens based on provider
        if max_tokens:
            self.max_tokens = max_tokens
        elif self.provider == "lmstudio":
            # For local models, we can use more tokens
            self.max_tokens = 8000  # Increased for local models
        else:
            self.max_tokens = MAX_TOKENS
            
        self.target_language = target_language or HTML_LANGUAGE
        
        # Set provider-specific options
        if self.provider == "openai":
            self.model = model or OPENAI_MODEL
            # Initialize OpenAI client
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key) if self.api_key else None
            except ImportError:
                logger.error("OpenAI package not installed. Please install it with 'pip install openai'")
                self.client = None
        elif self.provider == "lmstudio":
            self.model = model or LMSTUDIO_MODEL
            self.base_url = base_url or LMSTUDIO_BASE_URL
            self.client = "lmstudio"  # Mark as LM Studio for later checks
            
            # Parse max context length for LM Studio
            try:
                self.lmstudio_max_context = int(LMSTUDIO_MAX_CONTEXT_LENGTH)
                logger.info(f"LM Studio max context length: {self.lmstudio_max_context}")
            except ValueError:
                self.lmstudio_max_context = 16000  # Default fallback
                logger.warning(f"Invalid LM Studio max context length setting. Using default: {self.lmstudio_max_context}")
            
            # Log LM Studio configuration
            logger.info(f"Using LM Studio with model: {self.model}")
            logger.info(f"LM Studio base URL: {self.base_url}")
            logger.info(f"Max tokens setting: {self.max_tokens}")
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            self.client = None
        
        # Language name mapping
        self.language_names = {
            "it": "Italian",
            "en": "English",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            # Add more as needed
        }
        
        # Processing stats
        self.stats = {
            'articles_processed': 0,
            'articles_failed': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'chunks_processed': 0,
            'titles_deduplicated': 0,
            'content_truncated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'error_recoveries': 0
        }

        # Keep track of processed titles to avoid duplication
        self.processed_titles = set()
        
        # Set up caching
        if self.enable_caching:
            self.cache_dir = DOCS_DIR / ".cache" / "llm_responses"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"LLM response caching enabled. Cache directory: {self.cache_dir}")
        
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """
        Generate a cache key for a prompt and model.
        
        Args:
            prompt: The prompt for the LLM.
            model: The model used.
            
        Returns:
            A cache key string.
        """
        # Create a hash of the prompt and model
        key_data = f"{prompt}_{model}_{self.temperature}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, prompt: str, model: str) -> Optional[str]:
        """
        Try to get a response from the cache.
        
        Args:
            prompt: The prompt for the LLM.
            model: The model used.
            
        Returns:
            The cached response or None if not found.
        """
        if not self.enable_caching:
            return None
            
        cache_key = self._get_cache_key(prompt, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                    return cache_data['response']
            except Exception as e:
                logger.warning(f"Error reading from cache: {str(e)}")
                
        self.stats['cache_misses'] += 1
        return None
    
    def _save_to_cache(self, prompt: str, model: str, response: str) -> None:
        """
        Save a response to the cache.
        
        Args:
            prompt: The prompt for the LLM.
            model: The model used.
            response: The response to cache.
        """
        if not self.enable_caching:
            return
            
        cache_key = self._get_cache_key(prompt, model)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            cache_data = {
                'prompt': prompt,
                'model': model,
                'temperature': self.temperature,
                'response': response,
                'timestamp': time.time()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"Saved response to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
            
    def process_article_with_retries(self, article: Dict[str, Any], max_retries: int = 3, 
                                   progress_info: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Process an article with retries for improved reliability.
        
        Args:
            article: Article dictionary with content.
            max_retries: Maximum number of retry attempts.
            progress_info: Tuple of (current_item, total_items) for progress reporting.
            
        Returns:
            Updated article with translated and rewritten content.
        """
        for retry in range(max_retries):
            try:
                return self.process_article(article, progress_info)
            except Exception as e:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 2  # Exponential backoff
                    logger.warning(f"Error processing article: {str(e)}. Retrying in {wait_time}s ({retry+1}/{max_retries})")
                    time.sleep(wait_time)
                    self.stats['error_recoveries'] += 1
                else:
                    logger.error(f"Failed to process article after {max_retries} attempts: {str(e)}")
                    # Return the original article when all retries fail
                    return article

    def process_article(self, article: Dict[str, Any], progress_info: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Process an article by summarizing, rewriting, and translating it.
        
        Args:
            article: Article dictionary with content.
            progress_info: Tuple of (current_item, total_items) for progress reporting
            
        Returns:
            Updated article with translated and rewritten content.
        """
        if not self.client:
            logger.error("LLM client not initialized. Unable to process content.")
            return article
        
        processed_article = article.copy()
        start_time = time.time()
        
        try:
            # Get the content to process
            content = article.get('content', '') or article.get('summary', '')
            title = article.get('title', 'Untitled')
            
            # Log progress information if provided
            if progress_info:
                current_item, total_items = progress_info
                progress_percent = (current_item / total_items) * 100
                logger.info(f"Processing article {current_item} of {total_items} ({progress_percent:.1f}%): {title[:50]}...")
            
            if not content:
                logger.warning(f"No content to process for article: {title}")
                self.stats['articles_failed'] += 1
                return article
            
            # Process content based on length and provider
            content_length = len(content)
            
            # For LM Studio, use chunking for long content if context length allows
            if self.provider == "lmstudio":
                # Estimate tokens (rough approximation, 4 chars per token)
                estimated_tokens = content_length / 4
                
                # Reserve tokens for system message, prompt template, and response
                system_message_tokens = 50
                prompt_template_tokens = 200
                response_tokens = self.max_tokens
                
                # Calculate available tokens for content
                available_content_tokens = self.lmstudio_max_context - system_message_tokens - prompt_template_tokens - response_tokens
                
                # If content is too long even for chunking, use summary
                if estimated_tokens > available_content_tokens * 2:  # If even with chunking it's too big
                    logger.info(f"Content too long ({content_length} chars), generating summary...")
                    processed_content = self._generate_summary(article['title'], content)
                    logger.info(f"Summary generated: {len(processed_content)} chars")
                # If content can be processed in a single request or using chunking
                elif estimated_tokens <= available_content_tokens:
                    # Content fits in a single request
                    processed_content = content
                    logger.info(f"Content length ({content_length} chars) fits within LM Studio context window")
                else:
                    # Use chunking for long content
                    logger.info(f"Content length ({content_length} chars) requires chunking for LM Studio processing")
                    processed_content = self._process_content_in_chunks(article['title'], content)
            # For OpenAI, use their defaults (generate summary if too long)
            elif content_length > 1500 and self.provider == "openai":
                logger.info(f"Content too long ({content_length} chars), generating summary...")
                processed_content = self._generate_summary(article['title'], content)
                logger.info(f"Summary generated: {len(processed_content)} chars")
            else:
                processed_content = content
            
            # Detect language and skip translation if content is already in Italian
            detected_language = self._detect_language(processed_content)
            if detected_language == "it":
                logger.info(f"Content is already in Italian, skipping translation.")
                translated_title = title
                translated_content = processed_content
                # Track skipped translations
                if 'italian_skipped' in self.stats:
                    self.stats['italian_skipped'] += 1
                else:
                    self.stats['italian_skipped'] = 1
            else:
                # Translate and rewrite
                logger.info(f"Translating and rewriting article: {title[:50]}...")
                translated_title, translated_content = self._translate_and_rewrite(
                    article['title'], 
                    processed_content
                )
                logger.info(f"Translation complete: '{translated_title[:50]}...'")
            
            # Sanitize title and content
            translated_title = self._sanitize_title(translated_title)
            translated_content = self._sanitize_content(translated_content)
            
            # Update the article
            processed_article['original_title'] = article['title']
            processed_article['original_content'] = article.get('content', '')
            processed_article['title'] = translated_title
            processed_article['content'] = translated_content
            processed_article['language'] = self.target_language
            
            # Update stats
            self.stats['articles_processed'] += 1
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['avg_processing_time'] = self.stats['total_processing_time'] / self.stats['articles_processed']
            
            logger.info(f"Article processed in {processing_time:.2f} seconds")
            
            return processed_article
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing article {article.get('title', 'Unknown')}: {str(e)}")
            logger.error(f"Processing failed after {processing_time:.2f} seconds")
            self.stats['articles_failed'] += 1
            return article
    
    def process_topic_clusters(self, topic_clusters: List[Tuple[List[Dict[str, Any]], str]]) -> List[Tuple[List[Dict[str, Any]], str]]:
        """
        Process each cluster of articles.
        
        Args:
            topic_clusters: List of tuples containing (cluster, topic).
            
        Returns:
            Updated topic clusters with processed articles.
        """
        processed_clusters = []
        total_topics = len(topic_clusters)
        total_articles_to_process = sum(1 for cluster, _ in topic_clusters if cluster)
        
        logger.info(f"Starting to process {total_topics} topics with {total_articles_to_process} primary articles")
        logger.info(f"Using {self.provider} as LLM provider with {self.model}")
        
        # Reset stats
        self.stats = {
            'articles_processed': 0,
            'articles_failed': 0,
            'total_processing_time': 0,
            'avg_processing_time': 0,
            'italian_skipped': 0,
            'titles_deduplicated': 0,
            'content_truncated': 0
        }
        
        current_article = 0
        for i, (cluster, topic) in enumerate(topic_clusters):
            topic_progress = f"Topic {i+1}/{total_topics}"
            logger.info(f"\n{'-'*40}\n{topic_progress}: {topic}\n{'-'*40}")
            
            # Process the primary article in the cluster (the first one)
            if cluster:
                current_article += 1
                processed_primary = self.process_article(
                    cluster[0],
                    progress_info=(current_article, total_articles_to_process)
                )
                processed_cluster = [processed_primary] + cluster[1:]
                
                # Check if topic is already in Italian
                if self._detect_language(topic) == "it":
                    logger.info(f"Topic is already in Italian, skipping translation.")
                    translated_topic = topic
                    self.stats['italian_skipped'] += 1
                else:
                    # Translate the topic
                    logger.info(f"Translating topic: {topic}")
                    translated_topic = self._translate_text(topic)
                    logger.info(f"Topic translated: {translated_topic}")
                
                processed_clusters.append((processed_cluster, translated_topic))
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
        
        # Log final stats
        logger.info(f"\n{'-'*40}\nProcessing Summary\n{'-'*40}")
        logger.info(f"Completed processing {self.stats['articles_processed']} articles across {total_topics} topics")
        logger.info(f"Content already in Italian (skipped translation): {self.stats['italian_skipped']}")
        logger.info(f"Failed articles: {self.stats['articles_failed']}")
        logger.info(f"Total processing time: {self.stats['total_processing_time']:.2f} seconds")
        logger.info(f"Average time per article: {self.stats['avg_processing_time']:.2f} seconds")
        
        return processed_clusters
    
    def _generate_summary(self, title: str, content: str) -> str:
        """
        Generate a summary of the article content.
        
        Args:
            title: Article title.
            content: Article content.
            
        Returns:
            Summarized content.
        """
        # Truncate content if it's extremely long
        if len(content) > 10000:
            content = content[:10000] + "..."
            
        prompt = f"""Summarize the following news article in about 300-500 words, maintaining the key facts and information:

Title: {title}

Content: {content}

Summary:"""
        
        try:
            if self.provider == "openai":
                return self._call_openai_api(
                    system_message="You are a professional news editor specializing in concise summaries.",
                    user_message=prompt,
                    max_tokens=800  # Increased for better summaries
                )
            elif self.provider == "lmstudio":
                return self._call_lmstudio_api(
                    system_message="You are a professional news editor specializing in concise summaries.",
                    user_message=prompt,
                    max_tokens=800  # Increased for better summaries
                )
            else:
                return content[:1000] + "..."
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Return a truncated version if summarization fails
            return content[:1000] + "..."
    
    def _translate_and_rewrite(self, title: str, content: str) -> Tuple[str, str]:
        """
        Translate and rewrite the article in the target language.
        
        Args:
            title: Article title.
            content: Article content.
            
        Returns:
            Tuple of (translated_title, translated_content).
        """
        language_name = self.language_names.get(self.target_language, "Italian")
        
        prompt = f"""Translate and rewrite the following news article in {language_name}. 
Maintain journalistic tone and quality while adapting it for {language_name}-speaking readers.
Do not add any information not present in the original article.

Original Title: {title}

Original Content: {content}

Provide the response in the following format:
TITLE: [Translated title]

CONTENT: [Translated and rewritten content]"""
        
        try:
            system_message = f"You are a professional news translator specializing in {language_name}."
            
            logger.debug(f"Sending translation request to {self.provider} for {title}")
            start_time = time.time()
            
            # Check cache first
            cached_response = self._get_from_cache(prompt, self.model)
            if cached_response:
                result = cached_response
            else:
                if self.provider == "openai":
                    result = self._call_openai_api(
                        system_message=system_message,
                        user_message=prompt,
                        max_tokens=self.max_tokens
                    )
                elif self.provider == "lmstudio":
                    result = self._call_lmstudio_api(
                        system_message=system_message,
                        user_message=prompt,
                        max_tokens=self.max_tokens
                    )
                else:
                    return title, content
                
                # Save to cache
                self._save_to_cache(prompt, self.model, result)
            
            end_time = time.time()
            logger.debug(f"Translation completed in {end_time - start_time:.2f} seconds")
            
            # Extract title and content
            title_start = result.find("TITLE:") + 6 if "TITLE:" in result else 0
            content_start = result.find("CONTENT:") + 8 if "CONTENT:" in result else 0
            
            if content_start > 8:  # If the CONTENT marker was found
                translated_title = result[title_start:content_start].strip().replace("CONTENT:", "").strip()
                translated_content = result[content_start:].strip()
            else:
                # Fall back to splitting by newlines if markers aren't found
                parts = result.split("\n\n", 1)
                translated_title = parts[0].replace("TITLE:", "").strip()
                translated_content = parts[1].replace("CONTENT:", "").strip() if len(parts) > 1 else ""
            
            return translated_title, translated_content
        
        except Exception as e:
            logger.error(f"Error translating and rewriting: {str(e)}")
            return title, content
    
    def _translate_text(self, text: str) -> str:
        """
        Translate a piece of text to the target language.
        
        Args:
            text: Text to translate.
            
        Returns:
            Translated text.
        """
        language_name = self.language_names.get(self.target_language, "Italian")
        
        prompt = f"Translate the following text to {language_name}: {text}"
        
        try:
            system_message = f"You are a professional translator specializing in {language_name}."
            
            # Check cache first
            cached_response = self._get_from_cache(prompt, self.model)
            if cached_response:
                return cached_response
            
            if self.provider == "openai":
                result = self._call_openai_api(
                    system_message=system_message,
                    user_message=prompt,
                    max_tokens=200  # Increased for longer topics
                )
            elif self.provider == "lmstudio":
                result = self._call_lmstudio_api(
                    system_message=system_message,
                    user_message=prompt,
                    max_tokens=200  # Increased for longer topics
                )
            else:
                return text
            
            # Save to cache
            self._save_to_cache(prompt, self.model, result)
            return result
                
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return text
    
    def _call_openai_api(self, system_message: str, user_message: str, max_tokens: int = None) -> str:
        """
        Call the OpenAI API with the given messages.
        
        Args:
            system_message: System message for the chat.
            user_message: User message for the chat.
            max_tokens: Maximum tokens for generation.
            
        Returns:
            The generated response text.
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized")
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    def _call_lmstudio_api(self, system_message: str, user_message: str, max_tokens: int = None) -> str:
        """
        Call the LM Studio API with the given messages.
        
        Args:
            system_message: System message for the chat.
            user_message: User message for the chat.
            max_tokens: Maximum tokens for generation.
            
        Returns:
            The generated response text.
        """
        url = f"{self.base_url}/chat/completions"
        
        # Get tokens to use - local model supports -1 for unlimited
        tokens_to_use = -1 if max_tokens is None else max_tokens
        
        # Prepare the payload
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": self.temperature,
            "max_tokens": tokens_to_use,
            "stream": False
        }
        
        # Make the API request
        try:
            headers = {"Content-Type": "application/json"}
            logger.debug(f"Sending request to LM Studio with {len(user_message)} chars")
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)  # Increased timeout
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling LM Studio API: {str(e)}")
            raise
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to detect language for.
            
        Returns:
            Detected language code (e.g., 'en', 'it').
        """
        if not text or len(text) < 20:
            return "unknown"
            
        # Normalize text: lowercase and remove punctuation for more accurate matching
        text_lower = text.lower()
        
        # Common Italian words and patterns (articles, prepositions, conjunctions)
        italian_words = [
            "il", "lo", "la", "i", "gli", "le",  # articles
            "un", "uno", "una",  # indefinite articles
            "e", "ed", "o", "oppure", "ma", "però", "perché", "poiché",  # conjunctions
            "di", "a", "da", "in", "con", "su", "per", "tra", "fra",  # prepositions
            "che", "cui", "quale", "quali", "chi",  # relative pronouns
            "questo", "questa", "questi", "queste",  # demonstratives
            "sono", "è", "sei", "siamo", "siete", "hanno",  # common verb forms
            "non", "più", "meno", "molto", "poco", "troppo"  # adverbs
        ]
        
        # Italian-specific letter combinations and endings
        italian_patterns = [
            r"\b(?:ch|gh)\w+",  # Italian ch/gh patterns
            r"\w+(?:zione|tore|isti|ismo|ità)\b",  # common Italian endings
            r"\w+(?:tti|zza|lli|nno|cco|etto)\b",  # more Italian endings
            r"\b(?:dell[ao]|nell[ao]|all[ao]|dall[ao]|sull[ao])\b"  # articulated prepositions
        ]
        
        # Check if source is likely defined in metadata
        source_language = None
        if "italiana" in text_lower or "notizie italiane" in text_lower:
            source_language = "it"
            
        # Count Italian words
        word_count = sum(1 for word in italian_words if re.search(rf"\b{word}\b", text_lower))
        
        # Check for Italian patterns
        pattern_matches = sum(1 for pattern in italian_patterns if re.search(pattern, text_lower))
        
        # Calculate total score based on word count and patterns
        # Increase score weight for shorter texts
        text_length_factor = min(1.0, 500 / max(len(text), 100))
        score = (word_count * 2 + pattern_matches * 3) * text_length_factor
        
        # Define thresholds for language detection
        if source_language == "it" or score >= 6:
            return "it"
        else:
            return "other"  # Default to other if not confidently detected as Italian
    
    def _process_content_in_chunks(self, title: str, content: str, max_chunk_size: int = 8000) -> str:
        """
        Process long content by breaking it into chunks and processing each chunk separately.
        
        Args:
            title: Article title.
            content: Content to process.
            max_chunk_size: Maximum size of each chunk in characters.
            
        Returns:
            Processed content.
        """
        language_name = self.language_names.get(self.target_language, "Italian")
        
        # If content is short enough, process it directly
        if len(content) <= max_chunk_size:
            return self._translate_chunk(title, content, is_entire_article=True)
        
        logger.info(f"Processing article in chunks (total length: {len(content)} chars)")
        
        # Split content into paragraphs to maintain coherence
        paragraphs = re.split(r'\n\n+|\r\n\r\n+', content)
        
        # Group paragraphs into chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph) + 2  # +2 for newlines
            
            # If adding this paragraph would exceed the chunk size, finalize the current chunk
            if current_chunk and current_length + paragraph_length > max_chunk_size:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = paragraph_length
            else:
                current_chunk.append(paragraph)
                current_length += paragraph_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            total_chunks = len(chunks)
            
            # Include metadata about chunk position
            is_first_chunk = (i == 0)
            is_last_chunk = (i == len(chunks) - 1)
            is_entire_article = False
            
            logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} chars)")
            
            # Process the chunk
            processed_chunk = self._translate_chunk(
                title, 
                chunk, 
                chunk_num=chunk_num,
                total_chunks=total_chunks,
                is_first_chunk=is_first_chunk,
                is_last_chunk=is_last_chunk,
                is_entire_article=is_entire_article
            )
            
            processed_chunks.append(processed_chunk)
            self.stats['chunks_processed'] = self.stats.get('chunks_processed', 0) + 1
            
            # Add a small delay between chunks to avoid rate limits
            if i < len(chunks) - 1:
                time.sleep(0.5)
        
        # Combine processed chunks
        result = '\n\n'.join(processed_chunks)
        
        # Log completion
        logger.info(f"Completed processing {len(chunks)} chunks")
        
        return result
    
    def _translate_chunk(self, title: str, content: str, chunk_num: int = 1, total_chunks: int = 1, 
                        is_first_chunk: bool = True, is_last_chunk: bool = True, 
                        is_entire_article: bool = True) -> str:
        """
        Translate and rewrite a single chunk of content.
        
        Args:
            title: Article title.
            content: Chunk content to translate.
            chunk_num: Current chunk number.
            total_chunks: Total number of chunks.
            is_first_chunk: Whether this is the first chunk.
            is_last_chunk: Whether this is the last chunk.
            is_entire_article: Whether this chunk represents the entire article.
            
        Returns:
            Translated chunk content.
        """
        language_name = self.language_names.get(self.target_language, "Italian")
        
        # Construct a prompt based on whether we're processing a chunk or whole article
        if is_entire_article:
            prompt = f"""Translate and rewrite the following news article in {language_name}. 
Maintain journalistic tone and quality while adapting it for {language_name}-speaking readers.
Do not add any information not present in the original article.

Original Title: {title}

Original Content: {content}

Provide only the translated content without any preamble or explanation:"""
        else:
            # Build a prompt for a chunk with context about its position
            prompt = f"""Translate this chunk (part {chunk_num} of {total_chunks}) of a news article into {language_name}.
Maintain journalistic tone and quality. Translate only the content provided.

{'This is the beginning of the article.' if is_first_chunk else ''}
{'This is the end of the article.' if is_last_chunk else ''}

Original Title: {title}

Chunk {chunk_num}/{total_chunks}:
{content}

Provide only the translated content without any preamble, explanations, or mention of it being a chunk:"""
        
        system_message = f"You are a professional news translator specializing in {language_name}."
        
        try:
            logger.debug(f"Translating chunk {chunk_num}/{total_chunks} ({len(content)} chars)")
            
            # Check cache first
            cached_response = self._get_from_cache(prompt, self.model)
            if cached_response:
                translated_chunk = cached_response
            else:
                if self.provider == "openai":
                    translated_chunk = self._call_openai_api(
                        system_message=system_message,
                        user_message=prompt,
                        max_tokens=self.max_tokens
                    )
                elif self.provider == "lmstudio":
                    translated_chunk = self._call_lmstudio_api(
                        system_message=system_message,
                        user_message=prompt,
                        max_tokens=self.max_tokens
                    )
                else:
                    return content
                
                # Save to cache
                self._save_to_cache(prompt, self.model, translated_chunk)
            
            # Clean up the response to remove any chunk references
            translated_chunk = re.sub(r'^(Parte|Part|Chunk).*?[\n:]', '', translated_chunk, flags=re.IGNORECASE)
            translated_chunk = translated_chunk.strip()
            
            return translated_chunk
            
        except Exception as e:
            logger.error(f"Error translating chunk {chunk_num}/{total_chunks}: {str(e)}")
            return content
    
    def _sanitize_title(self, title: str) -> str:
        """
        Sanitize and deduplicate article titles.
        
        Args:
            title: The original title
            
        Returns:
            Clean, unique title
        """
        # Truncate if too long
        if len(title) > MAX_TITLE_LENGTH:
            title = title[:MAX_TITLE_LENGTH - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
            self.stats['content_truncated'] += 1
            
        # Remove duplicate colons or quotes that often appear in translated titles
        title = re.sub(r'["\']+', '"', title)
        title = re.sub(r'[:]+', ':', title)
        
        # Ensure uniqueness by adding a suffix if needed
        base_title = title
        counter = 1
        while title in self.processed_titles:
            counter += 1
            title = f"{base_title} ({counter})"
            self.stats['titles_deduplicated'] += 1
            
        self.processed_titles.add(title)
        return title
        
    def _sanitize_content(self, content: str, is_main_article: bool = False) -> str:
        """
        Sanitize and truncate content if needed.
        
        Args:
            content: The article content
            is_main_article: Whether this is the main (featured) article
            
        Returns:
            Sanitized content
        """
        if not content:
            return ""
            
        # Use different length limit for main articles
        max_length = MAX_MAIN_ARTICLE_LENGTH if is_main_article else MAX_CONTENT_LENGTH
            
        # Truncate if too long
        if len(content) > max_length:
            # Try to find a sentence end near the limit
            end_pos = content[:max_length].rfind('.')
            if end_pos > max_length * 0.8:  # If we can find a good breakpoint
                content = content[:end_pos+1] + TRUNCATION_SUFFIX
            else:
                content = content[:max_length - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
            self.stats['content_truncated'] += 1
            
        return content
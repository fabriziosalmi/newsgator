"""
LLM Integration Module

This module handles integration with LLMs for content translation and rewriting.
Supports both OpenAI and local LM Studio models.
"""

import logging
import os
import json
import requests
from typing import Dict, Any, List, Optional, Tuple
import time

from newsgator.config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, 
    LMSTUDIO_BASE_URL, LMSTUDIO_MODEL,
    LLM_TEMPERATURE, MAX_TOKENS, HTML_LANGUAGE
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
        target_language=None
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
        """
        self.provider = provider or LLM_PROVIDER
        self.api_key = api_key or OPENAI_API_KEY
        self.temperature = temperature or LLM_TEMPERATURE
        self.max_tokens = max_tokens or MAX_TOKENS
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
    
    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an article by summarizing, rewriting, and translating it.
        
        Args:
            article: Article dictionary with content.
            
        Returns:
            Updated article with translated and rewritten content.
        """
        if not self.client:
            logger.error("LLM client not initialized. Unable to process content.")
            return article
        
        processed_article = article.copy()
        
        try:
            # Get the content to process
            content = article.get('content', '') or article.get('summary', '')
            if not content:
                logger.warning(f"No content to process for article: {article.get('title', 'Unknown')}")
                return article
            
            # First generate a summary if content is too long
            if len(content) > 1500:
                summary = self._generate_summary(article['title'], content)
                processed_content = summary
            else:
                processed_content = content
            
            # Now translate and rewrite
            translated_title, translated_content = self._translate_and_rewrite(
                article['title'], 
                processed_content
            )
            
            # Update the article
            processed_article['original_title'] = article['title']
            processed_article['original_content'] = article.get('content', '')
            processed_article['title'] = translated_title
            processed_article['content'] = translated_content
            processed_article['language'] = self.target_language
            
            return processed_article
        
        except Exception as e:
            logger.error(f"Error processing article {article.get('title', 'Unknown')}: {str(e)}")
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
        
        for cluster, topic in topic_clusters:
            # Process the primary article in the cluster (the first one)
            if cluster:
                processed_primary = self.process_article(cluster[0])
                processed_cluster = [processed_primary] + cluster[1:]
                
                # Translate the topic
                translated_topic = self._translate_text(topic)
                
                processed_clusters.append((processed_cluster, translated_topic))
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
        
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
        prompt = f"""Summarize the following news article in about 300 words, maintaining the key facts and information:

Title: {title}

Content: {content}

Summary:"""
        
        try:
            if self.provider == "openai":
                return self._call_openai_api(
                    system_message="You are a professional news editor specializing in concise summaries.",
                    user_message=prompt,
                    max_tokens=400
                )
            elif self.provider == "lmstudio":
                return self._call_lmstudio_api(
                    system_message="You are a professional news editor specializing in concise summaries.",
                    user_message=prompt,
                    max_tokens=400
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
            
            if self.provider == "openai":
                return self._call_openai_api(
                    system_message=system_message,
                    user_message=prompt,
                    max_tokens=100
                )
            elif self.provider == "lmstudio":
                return self._call_lmstudio_api(
                    system_message=system_message,
                    user_message=prompt,
                    max_tokens=100
                )
            else:
                return text
                
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
        
        # Prepare the payload
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": self.temperature,
            "max_tokens": -1 if max_tokens is None else max_tokens,
            "stream": False
        }
        
        # Make the API request
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling LM Studio API: {str(e)}")
            raise
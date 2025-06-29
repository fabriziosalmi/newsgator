"""
Content Analysis Module

This module handles content analysis and clustering of similar articles.
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from newsgator.config import (
    SIMILARITY_THRESHOLD, MAX_ARTICLES_PER_CATEGORY,
    MAX_TITLE_LENGTH, MAX_CONTENT_LENGTH, MAX_SUMMARY_LENGTH, TRUNCATION_SUFFIX
)

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """Class for analyzing and grouping content based on similarity."""
    
    def __init__(self, similarity_threshold=None, max_articles_per_category=None):
        """
        Initialize the ContentAnalyzer.
        
        Args:
            similarity_threshold: Threshold for determining article similarity.
            max_articles_per_category: Maximum number of articles to keep per category.
        """
        self.similarity_threshold = similarity_threshold or SIMILARITY_THRESHOLD
        self.max_articles_per_category = max_articles_per_category or MAX_ARTICLES_PER_CATEGORY
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Changed from 2 to 1 to handle sparse data better
        )
        
        # Keep track of processed titles to avoid duplicates
        self.processed_titles = set()
    
    def cluster_articles(self, articles: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Cluster articles based on content similarity.
        
        Args:
            articles: List of article dictionaries.
            
        Returns:
            List of clusters, where each cluster is a list of similar articles.
        """
        if not articles:
            return []
        
        # Sanitize articles before clustering
        sanitized_articles = [self._sanitize_article(article) for article in articles]
        
        # Prepare article content for vectorization
        article_texts = []
        valid_indices = []
        
        for i, article in enumerate(sanitized_articles):
            # Combine title, summary, and content for better clustering
            text = f"{article.get('title', '')} {article.get('summary', '')} {article.get('content', '')[:1000]}"
            # Only include articles with sufficient text content
            if len(text.strip()) > 10:
                article_texts.append(text)
                valid_indices.append(i)
        
        # If no valid articles, return empty result
        if not article_texts:
            logger.warning("No valid article texts found for clustering")
            return [[a] for a in sanitized_articles]  # Each article as its own cluster
        
        valid_articles = [sanitized_articles[i] for i in valid_indices]
        
        try:
            # Vectorize the text content
            tfidf_matrix = self.vectorizer.fit_transform(article_texts)
            
            # Calculate similarity matrix (values between 0 and 1)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Ensure no negative values in the distance matrix
            # Convert similarity to distance (1 - similarity), clip to ensure no negative values
            distance_matrix = np.clip(1 - similarity_matrix, 0, 2)
            
            # Try DBSCAN first
            try:
                clustering = DBSCAN(
                    eps=1.0 - self.similarity_threshold,
                    min_samples=1,
                    metric='precomputed'
                ).fit(distance_matrix)
                
                labels = clustering.labels_
                
                # If all articles end up in the same cluster or every article is an outlier,
                # try hierarchical clustering instead
                unique_labels = set(labels)
                if len(unique_labels) <= 1 or (len(unique_labels) == len(valid_articles) and -1 in unique_labels):
                    raise ValueError("DBSCAN clustering produced poor results, trying hierarchical clustering")
                
            except Exception as e:
                logger.warning(f"DBSCAN clustering failed: {str(e)}. Trying hierarchical clustering.")
                
                # Fall back to hierarchical clustering
                n_clusters = min(int(len(valid_articles) / 2) + 1, 10)  # Reasonable number of clusters
                n_clusters = max(n_clusters, 2)  # At least 2 clusters if possible
                
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    linkage='average',
                    distance_threshold=None
                ).fit(distance_matrix)
                
                labels = clustering.labels_
            
            # Group articles by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(valid_articles[i])
            
            # Sort clusters by size (largest first)
            sorted_clusters = sorted(
                clusters.values(), 
                key=lambda x: len(x), 
                reverse=True
            )
            
            # Check if we missed any articles (those with invalid text)
            missed_articles = [a for i, a in enumerate(sanitized_articles) if i not in valid_indices]
            
            # Add any missed articles as their own clusters
            missed_clusters = [[a] for a in missed_articles]
            
            # Rank articles within each cluster
            result = [self._rank_articles(cluster) for cluster in sorted_clusters]
            result.extend(missed_clusters)
            
            return result
        
        except Exception as e:
            logger.error(f"Error clustering articles: {str(e)}")
            # Return each article as its own cluster if clustering fails
            return [[article] for article in sanitized_articles]
    
    def _rank_articles(self, cluster: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank articles within a cluster based on length and source priority.
        
        Args:
            cluster: A cluster of similar articles.
            
        Returns:
            Ranked list of articles, limited to max_articles_per_category.
        """
        # Simple ranking by content length and freshness
        for article in cluster:
            # Calculate a score based on content length and recency
            content_length = len(article.get('content', ''))
            summary_length = len(article.get('summary', ''))
            
            # Avoid division by zero or negative scores
            content_score = max(content_length * 0.8 + summary_length * 0.2, 1)
            
            # Boost score based on source reputation if needed
            # This could be enhanced with a source reputation dictionary
            source_boost = 1.0
            
            article['score'] = content_score * source_boost
        
        # Sort by score descending
        ranked_cluster = sorted(cluster, key=lambda x: x.get('score', 0), reverse=True)
        
        # Limit the number of articles per cluster
        return ranked_cluster[:self.max_articles_per_category]
    
    def _sanitize_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize article content and titles, ensuring they adhere to length limits.
        
        Args:
            article: The original article
            
        Returns:
            Sanitized article with proper length limits applied
        """
        sanitized = article.copy()
        
        # Sanitize title
        title = sanitized.get('title', '')
        if title and len(title) > MAX_TITLE_LENGTH:
            sanitized['title'] = title[:MAX_TITLE_LENGTH - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
        
        # Sanitize content
        content = sanitized.get('content', '')
        if content and len(content) > MAX_CONTENT_LENGTH:
            # Try to find a sentence end near the limit for cleaner truncation
            end_pos = content[:MAX_CONTENT_LENGTH].rfind('.')
            if end_pos > MAX_CONTENT_LENGTH * 0.8:
                sanitized['content'] = content[:end_pos+1] + TRUNCATION_SUFFIX
            else:
                sanitized['content'] = content[:MAX_CONTENT_LENGTH - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
        
        # Sanitize summary
        summary = sanitized.get('summary', '')
        if summary and len(summary) > MAX_SUMMARY_LENGTH:
            sanitized['summary'] = summary[:MAX_SUMMARY_LENGTH - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX
        
        return sanitized
    
    def extract_topics(self, clusters: List[List[Dict[str, Any]]]) -> List[Tuple[List[Dict[str, Any]], str]]:
        """
        Extract topics from clusters of similar articles.
        
        Args:
            clusters: List of article clusters.
            
        Returns:
            List of tuples, each containing (cluster, topic).
        """
        results = []
        existing_topics = set()  # Track topics to prevent duplicates
        
        for cluster in clusters:
            if not cluster:
                continue
                
            # Get the main article (first in cluster)
            main_article = cluster[0]
            
            # Extract topic from main article title
            topic = main_article.get('title', 'Untitled')
            
            # Ensure topic is unique
            base_topic = topic
            counter = 1
            while topic in existing_topics:
                counter += 1
                topic = f"{base_topic} ({counter})"
            
            existing_topics.add(topic)
            results.append((cluster, topic))
            
        return results
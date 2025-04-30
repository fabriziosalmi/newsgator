"""
Content Analysis Module

This module handles content analysis, including clustering similar articles
and identifying key topics.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

from newsgator.config import SIMILARITY_THRESHOLD, MAX_ARTICLES_PER_CATEGORY

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
            min_df=2,
        )
    
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
        
        # Prepare article content for vectorization
        article_texts = [
            f"{article['title']} {article['summary']} {article['content'][:1000]}"
            for article in articles
        ]
        
        try:
            # Vectorize the text content
            tfidf_matrix = self.vectorizer.fit_transform(article_texts)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Use DBSCAN for clustering
            clustering = DBSCAN(
                eps=1.0 - self.similarity_threshold,
                min_samples=1,
                metric='precomputed'
            ).fit(1 - similarity_matrix)
            
            # Get cluster labels
            labels = clustering.labels_
            
            # Group articles by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(articles[i])
            
            # Sort clusters by size (largest first)
            sorted_clusters = sorted(
                clusters.values(), 
                key=lambda x: len(x), 
                reverse=True
            )
            
            # Rank articles within each cluster
            return [self._rank_articles(cluster) for cluster in sorted_clusters]
        
        except Exception as e:
            logger.error(f"Error clustering articles: {str(e)}")
            # Return each article as its own cluster if clustering fails
            return [[article] for article in articles]
    
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
            content_score = len(article['content']) * 0.8 + len(article['summary']) * 0.2
            
            # Boost score based on source reputation if needed
            # This could be enhanced with a source reputation dictionary
            source_boost = 1.0
            
            article['score'] = content_score * source_boost
        
        # Sort by score descending
        ranked_cluster = sorted(cluster, key=lambda x: x.get('score', 0), reverse=True)
        
        # Limit the number of articles per cluster
        return ranked_cluster[:self.max_articles_per_category]
    
    def extract_topics(self, clusters: List[List[Dict[str, Any]]]) -> List[Tuple[List[Dict[str, Any]], str]]:
        """
        Extract main topics for each cluster.
        
        Args:
            clusters: List of article clusters.
            
        Returns:
            List of tuples containing (cluster, topic).
        """
        topics = []
        
        for cluster in clusters:
            if not cluster:
                continue
            
            # For simplicity, use the title of the highest-ranked article as the topic
            topic = cluster[0]['title']
            
            # More sophisticated topic extraction could be implemented here
            # For example, extracting key phrases from all articles in the cluster
            
            topics.append((cluster, topic))
        
        return topics
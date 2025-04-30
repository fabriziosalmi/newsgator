"""
HTML and RSS Generation Module

This module handles the generation of HTML files and RSS feeds for the processed news content.
"""

import logging
import os
from typing import List, Dict, Any, Tuple
from datetime import datetime
import json
from pathlib import Path
import shutil
import jinja2
import xml.etree.ElementTree as ET
from xml.dom import minidom

from newsgator.config import (
    HTML_TITLE, HTML_DESCRIPTION, HTML_AUTHOR, HTML_LANGUAGE, 
    TEMPLATES_DIR, DOCS_DIR, OUTPUT_RSS_FEED, RSS_FEED_TITLE,
    RSS_FEED_DESCRIPTION, RSS_FEED_LINK
)

logger = logging.getLogger(__name__)

class HTMLGenerator:
    """Class for generating HTML output and RSS feeds from processed articles."""
    
    def __init__(
        self, 
        output_dir=None, 
        templates_dir=None, 
        title=None, 
        description=None, 
        author=None, 
        language=None,
        generate_rss=None
    ):
        """
        Initialize the HTML Generator.
        
        Args:
            output_dir: Directory for output HTML files.
            templates_dir: Directory containing Jinja2 templates.
            title: Title for the generated HTML page.
            description: Description for the generated HTML page.
            author: Author for the generated HTML page.
            language: Language code for the generated HTML page.
            generate_rss: Whether to generate an RSS feed.
        """
        self.output_dir = Path(output_dir or DOCS_DIR)
        self.templates_dir = Path(templates_dir or TEMPLATES_DIR)
        self.title = title or HTML_TITLE
        self.description = description or HTML_DESCRIPTION
        self.author = author or HTML_AUTHOR
        self.language = language or HTML_LANGUAGE
        self.generate_rss = generate_rss if generate_rss is not None else OUTPUT_RSS_FEED
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self._init_jinja_env()
    
    def _init_jinja_env(self):
        """Initialize the Jinja2 environment with templates."""
        try:
            # Create templates directory if it doesn't exist
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            
            # Create default templates if they don't exist
            self._create_default_templates()
            
            # Set up Jinja2 environment
            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.templates_dir),
                autoescape=jinja2.select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
        except Exception as e:
            logger.error(f"Error initializing Jinja environment: {str(e)}")
            # Fallback to a very basic template system
            self.jinja_env = None
    
    def _create_default_templates(self):
        """Create default templates if they don't exist."""
        
        # Main template file
        main_template_path = self.templates_dir / "base.html"
        if not main_template_path.exists():
            with open(main_template_path, 'w', encoding='utf-8') as f:
                f.write("""<!DOCTYPE html>
<html lang="{{ language }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <meta name="description" content="{{ description }}">
    <meta name="author" content="{{ author }}">
    <link rel="stylesheet" href="css/styles.css">
    <!-- Add web fonts -->
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Source+Serif+Pro:wght@400;700&display=swap">
    <link rel="alternate" type="application/rss+xml" title="{{ title }} RSS Feed" href="feed.xml">
</head>
<body>
    <div class="newspaper">
        <header>
            <h1 class="newspaper-name">{{ title }}</h1>
            <div class="newspaper-info">
                <span class="date">{{ date }}</span>
                <span class="edition">Edizione Quotidiana</span>
            </div>
        </header>
        
        <main>
            {% for section in sections %}
            <section class="news-section">
                <h2 class="section-title">{{ section.title }}</h2>
                
                {% for article in section.articles %}
                <article class="news-article {% if loop.first %}lead-article{% endif %}" id="article-{{ section.id }}-{{ loop.index }}">
                    <h3 class="article-title">{{ article.title }}</h3>
                    {% if article.source %}
                    <div class="article-source">{{ article.source }}</div>
                    {% endif %}
                    <div class="article-content">
                        {{ article.content | safe }}
                    </div>
                </article>
                {% endfor %}
            </section>
            {% endfor %}
        </main>
        
        <footer>
            <p>&copy; {{ current_year }} {{ title }}. Generated by Newsgator.</p>
        </footer>
    </div>
</body>
</html>""")
        
        # CSS file
        css_dir = self.output_dir / "css"
        css_dir.mkdir(exist_ok=True)
        
        css_file_path = css_dir / "styles.css"
        if not css_file_path.exists():
            with open(css_file_path, 'w', encoding='utf-8') as f:
                f.write("""/* Newspaper-inspired styles */
:root {
    --newspaper-bg: #f8f7f2;
    --text-color: #222;
    --header-color: #111;
    --border-color: #ddd;
    --accent-color: #8b0000;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Source Serif Pro', 'Times New Roman', serif;
    line-height: 1.6;
    color: var(--text-color);
    background-color: #e0e0e0;
    padding: 20px;
}

.newspaper {
    max-width: 1200px;
    margin: 0 auto;
    background-color: var(--newspaper-bg);
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
    padding: 40px;
    border: 1px solid var(--border-color);
}

/* Header styles */
header {
    text-align: center;
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 20px;
    margin-bottom: 30px;
}

.newspaper-name {
    font-family: 'Playfair Display', 'Times New Roman', serif;
    font-size: 60px;
    font-weight: 900;
    letter-spacing: -1px;
    margin-bottom: 10px;
    color: var(--header-color);
}

.newspaper-info {
    display: flex;
    justify-content: space-between;
    font-style: italic;
    color: #555;
    font-size: 14px;
}

/* Section styles */
.news-section {
    margin-bottom: 40px;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 30px;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    border-bottom: 1px solid var(--accent-color);
    margin-bottom: 20px;
    padding-bottom: 8px;
}

/* Article styles */
.news-article {
    margin-bottom: 30px;
    column-count: 1;
}

.lead-article {
    column-count: 2;
    column-gap: 30px;
}

.article-title {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    margin-bottom: 10px;
    line-height: 1.3;
}

.lead-article .article-title {
    font-size: 32px;
    column-span: all;
}

.article-source {
    font-style: italic;
    font-size: 14px;
    margin-bottom: 10px;
    color: #555;
}

.article-content {
    font-size: 16px;
    text-align: justify;
    margin-bottom: 15px;
}

.article-content p {
    margin-bottom: 15px;
    text-indent: 1em;
}

/* Footer styles */
footer {
    text-align: center;
    font-size: 14px;
    color: #555;
    padding-top: 20px;
}

/* Media queries for responsive design */
@media screen and (max-width: 768px) {
    .newspaper {
        padding: 20px;
    }
    
    .newspaper-name {
        font-size: 40px;
    }
    
    .lead-article {
        column-count: 1;
    }
}""")
    
    def generate_html(self, processed_clusters: List[Tuple[List[Dict[str, Any]], str]]) -> str:
        """
        Generate HTML from processed article clusters.
        
        Args:
            processed_clusters: List of tuples containing (cluster, topic).
            
        Returns:
            Path to the generated HTML file.
        """
        if not processed_clusters:
            logger.warning("No content to generate HTML from")
            return ""
        
        try:
            # Prepare data for the template
            today = datetime.now().strftime("%d %B %Y")
            current_year = datetime.now().year
            
            # Organize data into sections
            sections = []
            for i, (cluster, topic) in enumerate(processed_clusters):
                if not cluster:
                    continue
                
                articles = []
                for article in cluster:
                    # Format the content for HTML
                    content_html = self._format_content(article.get('content', ''))
                    
                    articles.append({
                        'title': article.get('title', ''),
                        'content': content_html,
                        'source': article.get('source', ''),
                        'url': article.get('url', ''),
                        'published_date': article.get('published_date', ''),
                    })
                
                sections.append({
                    'title': topic,
                    'articles': articles,
                    'id': f"section-{i}"
                })
            
            # Save the articles data for potential reuse
            self._save_articles_data(processed_clusters)
            
            # Generate HTML using the template
            if self.jinja_env:
                template = self.jinja_env.get_template('base.html')
                html_content = template.render(
                    title=self.title,
                    description=self.description,
                    author=self.author,
                    language=self.language,
                    date=today,
                    current_year=current_year,
                    sections=sections
                )
                
                # Save the HTML file
                output_path = self.output_dir / "index.html"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                logger.info(f"HTML generated and saved to {output_path}")
                
                # Generate RSS feed if required
                if self.generate_rss:
                    rss_path = self.generate_rss_feed(processed_clusters)
                    logger.info(f"RSS feed generated and saved to {rss_path}")
                
                return str(output_path)
            else:
                logger.error("Jinja environment not initialized, cannot generate HTML")
                return ""
        
        except Exception as e:
            logger.error(f"Error generating HTML: {str(e)}")
            return ""
    
    def generate_rss_feed(self, processed_clusters: List[Tuple[List[Dict[str, Any]], str]]) -> str:
        """
        Generate an RSS feed for the processed articles.
        
        Args:
            processed_clusters: List of tuples containing (cluster, topic).
            
        Returns:
            Path to the generated RSS feed file.
        """
        try:
            # Create the RSS feed structure
            rss = ET.Element("rss", version="2.0")
            channel = ET.SubElement(rss, "channel")
            
            # Add channel metadata
            title = ET.SubElement(channel, "title")
            title.text = RSS_FEED_TITLE
            
            description = ET.SubElement(channel, "description")
            description.text = RSS_FEED_DESCRIPTION
            
            link = ET.SubElement(channel, "link")
            link.text = RSS_FEED_LINK
            
            language = ET.SubElement(channel, "language")
            language.text = self.language
            
            pub_date = ET.SubElement(channel, "pubDate")
            pub_date.text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
            
            generator = ET.SubElement(channel, "generator")
            generator.text = "Newsgator"
            
            # Add items for each article
            for i, (cluster, topic) in enumerate(processed_clusters):
                if not cluster:
                    continue
                
                # Take the first (main) article from each cluster
                article = cluster[0]
                
                item = ET.SubElement(channel, "item")
                
                item_title = ET.SubElement(item, "title")
                item_title.text = article.get('title', '')
                
                item_link = ET.SubElement(item, "link")
                # Create a link to the specific article in the HTML page
                item_link.text = f"{RSS_FEED_LINK}#article-section-{i}-1"
                
                item_description = ET.SubElement(item, "description")
                # Use the summary or truncated content
                content = article.get('summary', '') or article.get('content', '')[:300] + "..."
                item_description.text = content
                
                item_pub_date = ET.SubElement(item, "pubDate")
                if article.get('published_date'):
                    # Try to parse the ISO date and convert to RSS format
                    try:
                        pub_date_obj = datetime.fromisoformat(article['published_date'].replace('Z', '+00:00'))
                        item_pub_date.text = pub_date_obj.strftime("%a, %d %b %Y %H:%M:%S %z")
                    except (ValueError, TypeError):
                        item_pub_date.text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
                else:
                    item_pub_date.text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
                
                item_guid = ET.SubElement(item, "guid", isPermaLink="false")
                item_guid.text = f"newsgator-{i}-{datetime.now().strftime('%Y%m%d')}"
                
                if article.get('source'):
                    item_source = ET.SubElement(item, "source")
                    item_source.text = article['source']
            
            # Pretty print the XML
            rough_string = ET.tostring(rss, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent="  ")
            
            # Save the RSS feed
            output_path = self.output_dir / "feed.xml"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error generating RSS feed: {str(e)}")
            return ""
    
    def _format_content(self, content: str) -> str:
        """
        Format content for HTML display.
        
        Args:
            content: Raw article content.
            
        Returns:
            HTML-formatted content.
        """
        if not content:
            return ""
        
        # Split content into paragraphs and wrap in <p> tags
        paragraphs = content.split('\n\n')
        formatted = ''.join(f"<p>{p}</p>" for p in paragraphs if p.strip())
        
        return formatted
    
    def _save_articles_data(self, processed_clusters: List[Tuple[List[Dict[str, Any]], str]]) -> None:
        """
        Save the processed article data as JSON for potential reuse.
        
        Args:
            processed_clusters: List of tuples containing (cluster, topic).
        """
        try:
            data_path = self.output_dir / "data"
            data_path.mkdir(exist_ok=True)
            
            # Save data as JSON
            data_file = data_path / "articles.json"
            
            # Convert to serializable format
            serializable_data = []
            for cluster, topic in processed_clusters:
                cluster_data = {
                    'topic': topic,
                    'articles': []
                }
                
                for article in cluster:
                    # Create a simplified article representation
                    article_data = {
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'summary': article.get('summary', ''),
                        'source': article.get('source', ''),
                        'url': article.get('url', ''),
                        'language': article.get('language', self.language)
                    }
                    cluster_data['articles'].append(article_data)
                
                serializable_data.append(cluster_data)
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Article data saved to {data_file}")
        
        except Exception as e:
            logger.error(f"Error saving article data: {str(e)}")
    
    def publish_to_github_pages(self) -> bool:
        """
        Prepare content for GitHub Pages publishing.
        
        This function doesn't actually push to GitHub but ensures all
        necessary files are in the docs directory for GitHub Pages.
        
        Returns:
            True if preparation was successful, False otherwise.
        """
        try:
            # Ensure we have an index.html file
            if not (self.output_dir / "index.html").exists():
                logger.error("No index.html file found in output directory")
                return False
            
            # Create a .nojekyll file to prevent GitHub Pages from ignoring files
            # that start with an underscore
            nojekyll_path = self.output_dir / ".nojekyll"
            if not nojekyll_path.exists():
                with open(nojekyll_path, 'w') as f:
                    pass
            
            logger.info("Content prepared for GitHub Pages")
            return True
        
        except Exception as e:
            logger.error(f"Error preparing content for GitHub Pages: {str(e)}")
            return False
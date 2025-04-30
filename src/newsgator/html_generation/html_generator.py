"""
HTML Generation Module

This module generates HTML output and RSS feeds from processed articles.
"""

import logging
import os
import re
import math
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import jinja2

from newsgator.config import (
    DOCS_DIR, TEMPLATES_DIR, HTML_TITLE, HTML_DESCRIPTION, 
    HTML_AUTHOR, HTML_LANGUAGE, RSS_FEED_TITLE, RSS_FEED_DESCRIPTION, 
    RSS_FEED_LINK, OUTPUT_RSS_FEED, HTML_ARTICLES_PER_PAGE, HTML_MULTI_PAGE,
    HTML_PAGE_PREFIX, MAX_TITLE_LENGTH, MAX_CONTENT_LENGTH, MAX_MAIN_ARTICLE_LENGTH,
    TRUNCATION_SUFFIX
)

logger = logging.getLogger(__name__)

class HTMLGenerator:
    """Class for generating HTML output and RSS feeds from processed articles."""
    
    def __init__(self, output_dir=None, templates_dir=None):
        """
        Initialize the HTML Generator.
        
        Args:
            output_dir: Directory to save the generated files.
            templates_dir: Directory containing HTML templates.
        """
        self.output_dir = Path(output_dir or DOCS_DIR)
        self.templates_dir = Path(templates_dir or TEMPLATES_DIR)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Jinja environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Check if templates exist, create them if they don't
        if not (self.templates_dir / "base.html").exists():
            logger.info("Creating default templates")
            self._create_default_templates()
        
        # Load templates
        self.base_template = self.env.get_template("base.html")
        
        # Additional settings
        self.articles_per_page = HTML_ARTICLES_PER_PAGE
        self.multi_page = HTML_MULTI_PAGE
        self.page_prefix = HTML_PAGE_PREFIX
        
        logger.info(f"HTML Generator initialized with output directory: {self.output_dir}")
    
    def generate_html(self, topic_clusters: List[Tuple[List[Dict[str, Any]], str]]) -> str:
        """
        Generate HTML output from the processed topic clusters.
        
        Args:
            topic_clusters: List of tuples containing (cluster, topic).
            
        Returns:
            Path to the generated HTML file.
        """
        # Find the main article (article with most content)
        main_article, main_article_topic, main_article_idx = self._find_main_article(topic_clusters)
        
        # Create sections from topic clusters
        logger.info("Converting topic clusters to HTML sections")
        sections = self._create_sections(topic_clusters, main_article_idx)
        
        # Prepare the date display
        now = datetime.now()
        date_display = now.strftime("%-d %B %Y")  # e.g. "30 April 2025"
        
        # Generate table of contents
        toc = self._generate_table_of_contents(sections)
        
        # Prepare context for the template
        context = {
            "title": HTML_TITLE,
            "description": HTML_DESCRIPTION,
            "author": HTML_AUTHOR,
            "language": HTML_LANGUAGE,
            "date": date_display,
            "current_year": now.year,
            "toc": toc,
            "main_article": main_article,
            "main_article_topic": main_article_topic
        }
        
        # Generate multi-page content if enabled
        if self.multi_page and len(sections) > self.articles_per_page:
            logger.info(f"Generating multi-page content with {self.articles_per_page} sections per page")
            return self._generate_multipage_html(sections, context, main_article)
        else:
            # Generate single page content
            logger.info("Generating single page content")
            context["sections"] = sections
            html_content = self.base_template.render(**context)
            
            # Write the HTML to a file
            output_path = self.output_dir / "index.html"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            logger.info(f"HTML content written to {output_path}")
            return str(output_path)
    
    def _find_main_article(self, topic_clusters: List[Tuple[List[Dict[str, Any]], str]]) -> Tuple[Optional[Dict[str, Any]], str, int]:
        """
        Find the main article to feature on the front page.
        
        Args:
            topic_clusters: List of tuples containing (cluster, topic).
            
        Returns:
            Tuple of (main_article, topic, cluster_index).
        """
        main_article = None
        main_article_topic = ""
        main_article_idx = -1
        max_content_length = 0
        
        # Find the article with the most content
        for i, (cluster, topic) in enumerate(topic_clusters):
            if cluster:  # Skip empty clusters
                article = cluster[0]  # Primary article in the cluster
                content_length = len(article.get('content', ''))
                
                if content_length > max_content_length:
                    max_content_length = content_length
                    main_article = article
                    main_article_topic = topic
                    main_article_idx = i
        
        logger.info(f"Selected main article: '{main_article_topic}' with {max_content_length} characters")
        return main_article, main_article_topic, main_article_idx
    
    def _generate_multipage_html(self, sections: List[Dict[str, Any]], context: Dict[str, Any], main_article: Optional[Dict[str, Any]]) -> str:
        """
        Generate multi-page HTML content.
        
        Args:
            sections: List of section dictionaries.
            context: Base context dictionary.
            main_article: The main article to feature on the front page.
            
        Returns:
            Path to the main HTML file.
        """
        # Calculate the number of pages needed
        # First page needs fewer sections due to main article feature
        remaining_articles_first_page = max(0, self.articles_per_page - 1)
        total_other_sections = len(sections) - 1  # Excluding main article section
        
        # Calculate how many sections fit on first page (main article + other sections)
        first_page_section_count = 1 + remaining_articles_first_page
        remaining_sections = max(0, len(sections) - first_page_section_count)
        
        # Calculate pages needed for remaining sections after first page
        remaining_pages = math.ceil(remaining_sections / self.articles_per_page)
        
        num_pages = 1 + remaining_pages
        logger.info(f"Creating {num_pages} pages with main article featured on first page")
        
        # Generate pages
        page_paths = []
        
        # First page with main article
        first_page_section_indices = [0]  # Main article section always first
        if remaining_articles_first_page > 0:
            # Add other sections to first page, skipping the main article's section
            additional_indices = [i for i in range(1, len(sections)) if i < remaining_articles_first_page + 1]
            first_page_section_indices.extend(additional_indices)
            
        first_page_sections = [sections[i] for i in first_page_section_indices]
        
        # Generate first page with main article feature
        first_page_context = context.copy()
        first_page_context.update({
            "sections": first_page_sections,
            "current_page": 1,
            "total_pages": num_pages,
            "next_page": f"{self.page_prefix}2.html" if num_pages > 1 else None,
            "prev_page": None,
            "is_multi_page": True,
            "is_first_page": True,
            "has_main_article": main_article is not None
        })
        
        first_page_content = self.base_template.render(**first_page_context)
        first_page_path = self.output_dir / "index.html"
        with open(first_page_path, "w", encoding="utf-8") as f:
            f.write(first_page_content)
            
        page_paths.append(str(first_page_path))
        logger.info(f"Generated first page with main article feature")
        
        # Generate remaining pages
        if remaining_sections > 0:
            remaining_section_indices = [i for i in range(len(sections)) if i not in first_page_section_indices]
            
            for i in range(remaining_pages):
                page_num = i + 2  # Start from page 2
                start_idx = i * self.articles_per_page
                end_idx = min(start_idx + self.articles_per_page, len(remaining_section_indices))
                
                # Get indices for sections on this page
                page_indices = remaining_section_indices[start_idx:end_idx]
                
                # Get sections for this page
                page_sections = [sections[idx] for idx in page_indices]
                
                # Page context
                page_context = context.copy()
                page_context.update({
                    "sections": page_sections,
                    "current_page": page_num,
                    "total_pages": num_pages,
                    "prev_page": "index.html" if page_num == 2 else f"{self.page_prefix}{page_num-1}.html",
                    "next_page": f"{self.page_prefix}{page_num+1}.html" if page_num < num_pages else None,
                    "is_multi_page": True,
                    "is_first_page": False,
                    "has_main_article": False
                })
                
                # Generate page content
                page_content = self.base_template.render(**page_context)
                
                # Determine page filename
                page_filename = f"{self.page_prefix}{page_num}.html"
                    
                # Write the page
                page_path = self.output_dir / page_filename
                with open(page_path, "w", encoding="utf-8") as f:
                    f.write(page_content)
                    
                page_paths.append(str(page_path))
                logger.info(f"Generated page {page_num}/{num_pages}: {page_filename}")
        
        # Generate a simple pagination indicator for all pages
        self._add_pagination_to_pages(num_pages)
        
        return page_paths[0]  # Return path to the main index
    
    def _add_pagination_to_pages(self, num_pages: int) -> None:
        """
        Add pagination links to all generated pages.
        
        Args:
            num_pages: Total number of pages.
        """
        if num_pages <= 1:
            return
            
        # Generate a simple pagination script to be added to each page
        pagination_script = """
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Add pagination controls if not already present
                if (!document.querySelector('.pagination')) {
                    const paginationDiv = document.createElement('div');
                    paginationDiv.className = 'pagination';
                    paginationDiv.innerHTML = '"""
                    
        # Generate pagination HTML
        pagination_html = '<div class="page-numbers">'
        for i in range(1, num_pages + 1):
            if i == 1:
                page_url = 'index.html'
            else:
                page_url = f'{self.page_prefix}{i}.html'
                
            pagination_html += f'<a href="{page_url}" class="page-link" data-page="{i}">{i}</a>'
        
        pagination_html += '</div>'
        pagination_script += pagination_html.replace("'", "\\'")
        
        pagination_script += """';
                    
                    // Add before footer
                    const footer = document.querySelector('footer');
                    if (footer) {
                        footer.parentNode.insertBefore(paginationDiv, footer);
                    } else {
                        document.body.appendChild(paginationDiv);
                    }
                    
                    // Highlight current page
                    const currentPage = document.querySelector('meta[name="current-page"]')?.content;
                    if (currentPage) {
                        document.querySelector(`.page-link[data-page="${currentPage}"]`)?.classList.add('current');
                    }
                }
            });
        </script>
        """
        
        # Add the pagination script to all pages
        for i in range(num_pages):
            if i == 0:
                page_path = self.output_dir / "index.html"
            else:
                page_path = self.output_dir / f"{self.page_prefix}{i+1}.html"
                
            if page_path.exists():
                with open(page_path, "r+", encoding="utf-8") as f:
                    content = f.read()
                    # Insert meta tag for current page
                    head_end = content.find("</head>")
                    if head_end > 0:
                        meta_tag = f'<meta name="current-page" content="{i+1}">\n    '
                        content = content[:head_end] + meta_tag + content[head_end:]
                    
                    # Insert pagination script before body closes
                    body_end = content.rfind("</body>")
                    if body_end > 0:
                        content = content[:body_end] + pagination_script + content[body_end:]
                        
                    # Write updated content back
                    f.seek(0)
                    f.write(content)
                    f.truncate()
    
    def _generate_table_of_contents(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate a table of contents from sections.
        
        Args:
            sections: List of section dictionaries.
            
        Returns:
            Table of contents as a list of dictionaries.
        """
        toc = []
        for i, section in enumerate(sections):
            toc.append({
                "id": section["id"],
                "title": section["title"],
                "article_count": len(section["articles"]),
                "is_featured": section.get("is_featured", False)
            })
        return toc
    
    def generate_rss(self, topic_clusters: List[Tuple[List[Dict[str, Any]], str]]) -> Optional[str]:
        """
        Generate an RSS feed from the processed topic clusters.
        
        Args:
            topic_clusters: List of tuples containing (cluster, topic).
            
        Returns:
            Path to the generated RSS file, or None if RSS generation is disabled.
        """
        if not OUTPUT_RSS_FEED:
            logger.info("RSS feed generation is disabled")
            return None
            
        logger.info("Generating RSS feed")
        
        # Create the root element
        rss = ET.Element("rss")
        rss.set("version", "2.0")
        
        # Create the channel element
        channel = ET.SubElement(rss, "channel")
        
        # Add channel metadata
        ET.SubElement(channel, "title").text = RSS_FEED_TITLE
        ET.SubElement(channel, "description").text = RSS_FEED_DESCRIPTION
        ET.SubElement(channel, "link").text = RSS_FEED_LINK
        ET.SubElement(channel, "language").text = HTML_LANGUAGE
        ET.SubElement(channel, "pubDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S ")
        ET.SubElement(channel, "generator").text = "Newsgator"
        
        # Add items from topic clusters
        for i, (cluster, topic) in enumerate(topic_clusters):
            if cluster:  # Only process non-empty clusters
                # Use the first (most important) article in each cluster
                article = cluster[0]
                
                # Create item element
                item = ET.SubElement(channel, "item")
                
                # Add item metadata
                ET.SubElement(item, "title").text = article.get("title", topic)
                
                # Create a link to the article in the HTML
                article_link = f"{RSS_FEED_LINK}#article-section-{i}-1"
                ET.SubElement(item, "link").text = article_link
                
                # Add description (summary or truncated content)
                description = article.get("summary", "") or article.get("content", "")[:500]
                ET.SubElement(item, "description").text = description
                
                # Add publication date if available, otherwise use current time
                pub_date = article.get("published", datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000"))
                ET.SubElement(item, "pubDate").text = pub_date
                
                # Add unique GUID
                guid = ET.SubElement(item, "guid")
                guid.text = f"newsgator-{i}-{datetime.now().strftime('%Y%m%d')}"
                guid.set("isPermaLink", "false")
                
                # Add source if available
                if "source" in article:
                    ET.SubElement(item, "source").text = article["source"]
        
        # Format the XML with pretty printing
        rough_string = ET.tostring(rss, "utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # Write the RSS to a file
        output_path = self.output_dir / "feed.xml"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)
            
        logger.info(f"RSS feed written to {output_path}")
        return str(output_path)
    
    def _create_sections(self, topic_clusters: List[Tuple[List[Dict[str, Any]], str]], main_article_idx: int = -1) -> List[Dict[str, Any]]:
        """
        Convert topic clusters to sections for HTML generation.
        
        Args:
            topic_clusters: List of tuples containing (cluster, topic).
            main_article_idx: Index of the main article to feature.
            
        Returns:
            List of section dictionaries.
        """
        sections = []
        
        for i, (cluster, topic) in enumerate(topic_clusters):
            if not cluster:  # Skip empty clusters
                continue
                
            # Create section ID
            section_id = f"section-{i}"
            
            # Mark if this is the featured section
            is_featured = (i == main_article_idx)
            
            # Create article objects for the section
            articles = []
            for j, article in enumerate(cluster):
                article_id = f"article-{section_id}-{j+1}"
                is_main_article = (is_featured and j == 0)
                
                # Add image if available
                image_path = article.get("image_path")
                image_caption = f"Image for {article.get('title', 'article')}"
                
                # Extract a pull quote for main articles
                pull_quote = None
                if is_main_article and article.get('content'):
                    pull_quote = self._extract_pull_quote(article['content'])
                
                articles.append({
                    "id": article_id,
                    "title": article.get("title", "Untitled"),
                    "content": article.get("content", ""),
                    "source": article.get("source", ""),
                    "url": article.get("url", ""),
                    "image_path": image_path,
                    "image_caption": image_caption,
                    "is_main_article": is_main_article,
                    "pull_quote": pull_quote
                })
            
            # Create the section
            sections.append({
                "id": section_id,
                "title": topic,
                "articles": articles,
                "is_featured": is_featured
            })
        
        return sections
    
    def _extract_pull_quote(self, content: str, min_length: int = 60, max_length: int = 120) -> Optional[str]:
        """
        Extract a pull quote from article content.
        
        Args:
            content: Article content.
            min_length: Minimum length of pull quote.
            max_length: Maximum length of pull quote.
            
        Returns:
            A good pull quote or None if not found.
        """
        if not content or len(content) < min_length:
            return None
            
        # Split content into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Filter sentences by length
        good_sentences = [s for s in sentences if min_length <= len(s) <= max_length and '.' in s and '"' not in s]
        
        # If no good sentences, try to find something close
        if not good_sentences and sentences:
            good_sentences = [s for s in sentences if len(s) <= max_length * 1.2 and s.strip()]
        
        # Return a random sentence from the middle third of the article if possible
        if good_sentences:
            if len(good_sentences) > 3:
                # Pick from middle third
                start_idx = len(good_sentences) // 3
                end_idx = start_idx * 2
                candidates = good_sentences[start_idx:end_idx]
                return random.choice(candidates)
            else:
                return random.choice(good_sentences)
                
        return None
    
    def _create_default_templates(self):
        """Create default templates if they don't exist."""
        # Create the templates directory if it doesn't exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Base HTML template
        base_html = """<!DOCTYPE html>
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
    {% if is_multi_page %}
    <meta name="current-page" content="{{ current_page }}">
    {% endif %}
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
        
        {% if is_multi_page %}
        <div class="pagination-info">
            <span class="page-indicator">Pagina {{ current_page }} di {{ total_pages }}</span>
            <div class="page-navigation">
                {% if prev_page %}
                <a href="{{ prev_page }}" class="page-nav prev-page">&laquo; Pagina Precedente</a>
                {% endif %}
                {% if next_page %}
                <a href="{{ next_page }}" class="page-nav next-page">Pagina Successiva &raquo;</a>
                {% endif %}
            </div>
        </div>
        {% endif %}
        
        {% if toc and is_multi_page %}
        <div class="table-of-contents">
            <h2 class="toc-header">In Questa Edizione</h2>
            <ul class="toc-list">
                {% for item in toc %}
                <li class="toc-item">
                    <a href="#{{ item.id }}" class="toc-link">{{ item.title }}</a>
                </li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <main>
            {% if has_main_article %}
            <section class="main-article-section">
                <h2 class="main-article-title">{{ main_article_topic }}</h2>
                <article class="main-article" id="main-article">
                    <h3 class="article-title">{{ main_article.title }}</h3>
                    {% if main_article.source %}
                    <div class="article-source">{{ main_article.source }}</div>
                    {% endif %}
                    
                    {% if main_article.image_path %}
                    <div class="article-image">
                        <img src="{{ main_article.image_path }}" alt="{{ main_article.image_caption }}" class="newspaper-image">
                        <div class="image-caption">{{ main_article.image_caption }}</div>
                    </div>
                    {% endif %}
                    
                    <div class="article-content">
                        {{ main_article.content | safe }}
                    </div>
                    
                    {% if main_article.pull_quote %}
                    <div class="pull-quote">{{ main_article.pull_quote }}</div>
                    {% endif %}
                </article>
            </section>
            {% endif %}
            
            {% for section in sections %}
            <section class="news-section" id="{{ section.id }}">
                <h2 class="section-title">{{ section.title }}</h2>
                
                {% for article in section.articles %}
                <article class="news-article {% if loop.first %}lead-article{% endif %}" id="{{ article.id }}">
                    <h3 class="article-title">{{ article.title }}</h3>
                    {% if article.source %}
                    <div class="article-source">{{ article.source }}</div>
                    {% endif %}
                    
                    {% if article.image_path and loop.first %}
                    <div class="article-image">
                        <img src="{{ article.image_path }}" alt="{{ article.image_caption }}" class="newspaper-image">
                        <div class="image-caption">{{ article.image_caption }}</div>
                    </div>
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
</html>"""
        
        with open(self.templates_dir / "base.html", "w", encoding="utf-8") as f:
            f.write(base_html)
            
        # Create CSS directory and default styles
        css_dir = self.output_dir / "css"
        css_dir.mkdir(parents=True, exist_ok=True)
        
        css_content = """/* Newspaper-inspired styles */
:root {
    --newspaper-bg: #f8f7f2;
    --text-color: #222;
    --header-color: #111;
    --border-color: #ddd;
    --accent-color: #8b0000;
    --link-color: #0056b3;
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

/* Table of Contents */
.table-of-contents {
    margin-bottom: 30px;
    padding: 15px;
    background-color: #f1f1f1;
    border: 1px solid var(--border-color);
}

.toc-header {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    margin-bottom: 10px;
    text-align: center;
}

.toc-list {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    list-style-type: none;
}

.toc-item {
    flex: 0 0 48%;
    margin-bottom: 5px;
}

.toc-link {
    color: var(--link-color);
    text-decoration: none;
    font-weight: 500;
}

.toc-link:hover {
    text-decoration: underline;
}

/* Pagination styles */
.pagination-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-color);
}

.page-indicator {
    font-weight: bold;
    font-family: 'Playfair Display', serif;
}

.page-navigation {
    display: flex;
    gap: 15px;
}

.page-nav {
    color: var(--link-color);
    text-decoration: none;
    font-weight: 500;
}

.page-nav:hover {
    text-decoration: underline;
}

.pagination {
    text-align: center;
    margin: 20px 0;
    padding: 10px 0;
    border-top: 1px solid var(--border-color);
}

.page-numbers {
    display: flex;
    justify-content: center;
    gap: 5px;
}

.page-link {
    display: inline-block;
    padding: 5px 10px;
    text-decoration: none;
    color: var(--text-color);
    border: 1px solid var(--border-color);
}

.page-link:hover {
    background-color: #f0f0f0;
}

.page-link.current {
    background-color: var(--accent-color);
    color: white;
    font-weight: bold;
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
    column-span: all;
}

.lead-article .article-title {
    font-size: 32px;
}

.article-source {
    font-style: italic;
    font-size: 14px;
    margin-bottom: 10px;
    color: #555;
    column-span: all;
}

/* Image styles */
.article-image {
    margin: 0 0 20px 0;
    column-span: all;
    text-align: center;
}

.newspaper-image {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto 8px;
    border: 1px solid var(--border-color);
    filter: grayscale(100%);
}

.image-caption {
    font-style: italic;
    font-size: 14px;
    color: #555;
    text-align: center;
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

/* Pull quote styles */
.pull-quote {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    line-height: 1.4;
    font-style: italic;
    color: var(--accent-color);
    padding: 15px 0;
    margin: 20px 0;
    border-top: 1px solid var(--border-color);
    border-bottom: 1px solid var(--border-color);
    text-align: center;
}

/* Main article section */
.main-article-section {
    margin-bottom: 50px;
    border-bottom: 2px solid var(--accent-color);
    padding-bottom: 40px;
}

.main-article-title {
    font-family: 'Playfair Display', serif;
    font-size: 36px;
    margin-bottom: 20px;
    text-align: center;
    font-weight: 900;
}

.main-article {
    column-count: 2;
    column-gap: 40px;
}

.main-article .article-title {
    font-size: 32px;
    margin-bottom: 15px;
    line-height: 1.2;
}

/* Footer styles */
footer {
    margin-top: 40px;
    padding-top: 20px;
    text-align: center;
    font-size: 14px;
    color: #555;
    border-top: 1px solid var(--border-color);
}

/* Media Queries */
@media (max-width: 768px) {
    .newspaper {
        padding: 20px;
    }
    
    .newspaper-name {
        font-size: 40px;
    }
    
    .section-title {
        font-size: 24px;
    }
    
    .article-title {
        font-size: 20px;
    }
    
    .lead-article .article-title, 
    .main-article .article-title {
        font-size: 26px;
    }
    
    .main-article, 
    .lead-article {
        column-count: 1;
    }
    
    .pull-quote {
        font-size: 20px;
    }
    
    .main-article-title {
        font-size: 28px;
    }
    
    .toc-list {
        flex-direction: column;
    }
    
    .toc-item {
        flex: 0 0 100%;
    }
}
"""
        
        with open(self.output_dir / "css" / "styles.css", "w", encoding="utf-8") as f:
            f.write(css_content)
# Newsgator

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://github.com/fabriziosalmi/newsgator)

**An intelligent RSS feed aggregator that transforms news into beautiful newspaper-style publications**

Newsgator is an advanced RSS feed aggregator that collects news from various sources, analyzes content for similarity, uses LLM to translate and rewrite content in Italian, and publishes beautifully styled newspaper-like HTML and RSS feeds.

## 📖 Table of Contents

- [Demo](#-demo)
- [Features](#features)
- [Quick Start](#-quick-start)
- [Prerequisites](#-prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [GitHub Pages Publishing](#github-pages-publishing)
- [Customization](#customization)
- [Project Structure](#project-structure)
- [Performance & Limitations](#-performance--limitations)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)
- [Contributing](#contributing)
- [License](#license)

## 🎬 Demo

Newsgator transforms RSS feeds into beautiful, newspaper-style web pages:

![Newsgator Demo](https://github.com/user-attachments/assets/e1149003-1667-4c35-bff9-6e8b713dcc56)

*Example of generated newspaper-style HTML output with Italian news articles*

## Features

- **🔄 RSS Feed Collection**: Fetches and parses RSS feeds from multiple news sources
- **🧠 Content Analysis**: Groups similar articles using natural language processing techniques
- **🌐 LLM Translation & Rewriting**: Translates content to Italian and rewrites it in a journalistic style
  - Supports both OpenAI models and local LM Studio models (including phi-4-mini-instruct)
- **📰 Newspaper-Style HTML**: Generates beautifully formatted HTML with a classic newspaper design
- **📡 RSS Feed Generation**: Creates an RSS feed of translated and processed articles
- **🐳 Docker Support**: Run in a container with a built-in web server to view the content

## 🚀 Quick Start

Get Newsgator running in under 5 minutes:

```bash
# Clone the repository
git clone https://github.com/fabriziosalmi/newsgator.git
cd newsgator

# Option 1: Using Docker (recommended)
docker build -t newsgator .
docker run -p 8080:8080 newsgator
# Visit http://localhost:8080 to see your newspaper!

# Option 2: Install locally
pip install -e .
python main.py
# Open docs/index.html in your browser
```

> **Note**: For LLM functionality, you'll need either OpenAI API access or a local LM Studio instance running.

## 📋 Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: At least 2GB RAM (4GB recommended for large feeds)
- **Storage**: 500MB free space for dependencies and generated content
- **Network**: Internet connection for RSS feeds and LLM API calls

### LLM Requirements (choose one)

**Option A: OpenAI API** (easier setup)
- OpenAI API key with sufficient credits
- Models supported: GPT-4, GPT-3.5-turbo, or newer

**Option B: Local LM Studio** (free, privacy-focused)
- [LM Studio](https://lmstudio.ai/) installed and running
- phi-4-mini-instruct model downloaded (or compatible model)
- At least 8GB RAM for model inference

### Optional

- **Docker**: For containerized deployment (recommended)
- **Git**: For version control and GitHub Pages publishing

## Installation

### Option 1: Install from source

1. Clone the repository:
   ```
   git clone https://github.com/fabriziosalmi/newsgator.git
   cd newsgator
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

### Option 2: Using Docker

1. Clone the repository:
   ```
   git clone https://github.com/fabriziosalmi/newsgator.git
   cd newsgator
   ```

2. Build the Docker image:
   ```
   docker build -t newsgator .
   ```

3. Run the container (make sure LM Studio is running locally first):
   ```
   docker run -p 8080:8080 newsgator
   ```

   Or with OpenAI:
   ```
   docker run -p 8080:8080 -e LLM_PROVIDER="openai" -e OPENAI_API_KEY="your-api-key-here" newsgator
   ```

## Configuration

### LLM Options

Newsgator supports two options for the LLM provider:

1. **LM Studio (Default)**: Uses a local LM Studio instance with phi-4-mini-instruct
   - Make sure to have LM Studio running locally at http://localhost:1234
   - The phi-4-mini-instruct model should be loaded in LM Studio
   - Set environment variables if needed:
     ```
     export LLM_PROVIDER="lmstudio"
     export LMSTUDIO_BASE_URL="http://localhost:1234/v1"
     export LMSTUDIO_MODEL="phi-4-mini-instruct"
     ```

2. **OpenAI**: Uses OpenAI's models (requires an API key)
   - Set environment variables:
     ```
     export LLM_PROVIDER="openai"
     export OPENAI_API_KEY="your-api-key-here"
     export OPENAI_MODEL="gpt-4"  # or another model
     ```

Additional settings:
- Customize feeds and other settings in `src/newsgator/config.py` as needed.

## Usage

### Basic Usage

Run the application to fetch feeds, process content, and generate HTML and RSS output:

```
python main.py
```

The generated content will be placed in the `docs/` directory, which you can:
- View locally
- Push to GitHub Pages
- Serve with a web server

### Docker Usage

The Docker container supports different operation modes:

1. Generate content once and serve it (with LM Studio):
   ```
   # Make sure LM Studio is running locally first
   docker run -p 8080:8080 newsgator
   ```

2. Generate content once and serve it (with OpenAI):
   ```
   docker run -p 8080:8080 -e LLM_PROVIDER="openai" -e OPENAI_API_KEY="your-api-key" newsgator
   ```

3. Only generate content (without serving):
   ```
   docker run -e LLM_PROVIDER="lmstudio" newsgator --mode generate
   ```

4. Only serve existing content:
   ```
   docker run -p 8080:8080 newsgator --mode serve
   ```

5. Generate content periodically (every 24 hours) and serve it:
   ```
   docker run -p 8080:8080 -e LLM_PROVIDER="lmstudio" newsgator --mode both --interval 24
   ```

### Docker Networking Note

When running in Docker, the container will try to connect to LM Studio at `host.docker.internal:1234` which maps to your host machine's localhost. This should work automatically on Docker Desktop for Mac and Windows. For Linux, you may need to add `--add-host=host.docker.internal:host-gateway` to your docker run command.

## GitHub Pages Publishing

After generating content in the `docs/` directory, you can manually push to GitHub:

1. Commit the changes in the `docs/` directory:
   ```
   git add docs/
   git commit -m "Update news content"
   git push
   ```

2. In your GitHub repository, go to Settings → Pages
3. Set Source to "Deploy from a branch"
4. Select the `main` branch and `/docs` folder
5. Click Save

## Customization

- **RSS Feeds**: Edit the `RSS_FEEDS` list in `config.py` to add or remove sources
- **HTML Design**: Modify the templates in `src/newsgator/templates/` or the CSS in `docs/css/styles.css`
- **Content Analysis Settings**: Adjust similarity thresholds and clustering parameters in `config.py`
- **LLM Settings**: Change target language, model, or temperature in `config.py`

## Project Structure

```
newsgator/
├── docs/                  # Output directory (HTML, RSS, CSS)
├── src/
│   └── newsgator/         # Main package
│       ├── feed_processing/      # RSS feed fetching and parsing
│       ├── content_analysis/     # Content similarity and clustering
│       ├── llm_integration/      # LLM translation and rewriting
│       ├── html_generation/      # HTML and RSS generation
│       ├── web_server/           # Simple HTTP server for Docker
│       └── templates/            # Jinja2 templates
├── docker-entrypoint.py   # Docker entry point script
├── Dockerfile             # Docker configuration
├── main.py                # Entry point script
├── requirements.txt       # Python dependencies
├── setup.py               # Package setup file
└── README.md              # This file
```

## ⚡ Performance & Limitations

### Processing Time
- **Small feeds** (5-10 articles): ~2-5 minutes
- **Medium feeds** (20-30 articles): ~5-15 minutes  
- **Large feeds** (50+ articles): ~15-30 minutes

*Processing time depends on LLM provider, article length, and system performance.*

### Current Limitations
- **Language**: Currently optimized for Italian translation only
- **Article limits**: Maximum of 5 articles per category by default
- **Feed sources**: Pre-configured Italian news sources (customizable)
- **LLM context**: Limited by model's maximum context length (32K tokens for phi-4)
- **Rate limiting**: Subject to OpenAI API rate limits when using OpenAI

### Resource Usage
- **Memory**: ~1-2GB during processing
- **Storage**: Generated content typically 10-50MB per run
- **Network**: Downloads RSS feeds and makes LLM API calls

## 🔧 Troubleshooting

### Common Issues

**"No articles fetched" error**
```bash
# Check internet connection and RSS feed URLs
curl -I https://www.ansa.it/sito/notizie/topnews/topnews_rss.xml

# Verify config.yaml has valid RSS feeds
cat config.yaml | grep -A 20 rss_feeds
```

**LM Studio connection failed**
```bash
# Ensure LM Studio is running on correct port
curl http://localhost:1234/v1/models

# Check if phi-4-mini-instruct model is loaded
# Open LM Studio GUI and verify model status
```

**Docker container networking issues**
```bash
# For Linux systems, add host networking
docker run --add-host=host.docker.internal:host-gateway -p 8080:8080 newsgator

# Alternative: Use host network mode
docker run --network host newsgator
```

**Out of memory errors**
- Reduce `max_items_per_feed` in config.yaml
- Use a smaller LLM model
- Increase system swap space

**Permission denied writing to docs/**
```bash
# Fix permissions
chmod 755 docs/
sudo chown -R $USER:$USER docs/
```

### Debug Mode

Enable debug logging for more detailed information:

```bash
python main.py --debug
```

## ❓ FAQ

**Q: Can I use languages other than Italian?**
A: Currently, Newsgator is optimized for Italian translation. You can modify the LLM prompts in the source code to target other languages.

**Q: How much does it cost to run with OpenAI?**
A: Costs vary based on article volume and model choice. Typical usage: $0.50-$2.00 per run with GPT-4, $0.10-$0.50 with GPT-3.5-turbo.

**Q: Can I add my own RSS feeds?**
A: Yes! Edit the `rss_feeds` section in `config.yaml` to add your preferred news sources.

**Q: How often should I run Newsgator?**
A: For daily news: once per day. For real-time updates: every 2-4 hours. Consider API rate limits and costs.

**Q: Can I customize the newspaper design?**
A: Yes! Modify the templates in `src/newsgator/templates/` and CSS in `docs/css/styles.css`.

**Q: Is my data private when using local LM Studio?**
A: Yes! With LM Studio, all processing happens locally. No data is sent to external services except for RSS feed fetching.

**Q: Can I run this on a Raspberry Pi?**
A: Yes, but LM Studio requires significant resources. Consider using OpenAI API instead for lightweight deployments.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

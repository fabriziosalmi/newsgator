# Newsgator

Newsgator is an advanced RSS feed aggregator that collects news from various sources, analyzes content for similarity, uses LLM to translate and rewrite content in Italian, and publishes beautifully styled newspaper-like HTML and RSS feeds.

## Features

- **RSS Feed Collection**: Fetches and parses RSS feeds from multiple news sources
- **Content Analysis**: Groups similar articles using natural language processing techniques
- **LLM Translation & Rewriting**: Translates content to Italian and rewrites it in a journalistic style
  - Supports both OpenAI models and local LM Studio models (including phi-4-mini-instruct)
- **Newspaper-Style HTML**: Generates beautifully formatted HTML with a classic newspaper design
- **RSS Feed Generation**: Creates an RSS feed of translated and processed articles
- **Docker Support**: Run in a container with a built-in web server to view the content

## Requirements

- Python 3.8+
- Either:
  - OpenAI API key (for using OpenAI models), OR
  - Local LM Studio instance running with phi-4-mini-instruct model (default)
- Docker (optional, for containerized deployment)

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

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

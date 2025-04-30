FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set up environment
ENV PYTHONPATH=/app
ENV LLM_PROVIDER="lmstudio"
ENV LMSTUDIO_BASE_URL="http://127.0.0.1:1234/v1"
ENV LMSTUDIO_MODEL="phi-4-mini-instruct"
ENV OPENAI_API_KEY=""

# Install app in development mode
RUN pip install -e .

# Create a directory for output
RUN mkdir -p /app/docs

# Create a simple web server module
RUN mkdir -p /app/src/newsgator/web_server

# Expose ports for the web server
EXPOSE 8080

# Set the entrypoint
ENTRYPOINT ["python", "docker-entrypoint.py"]
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/knowledge_base \
    models/embeddings metrics

# Expose Streamlit port
EXPOSE 8501

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
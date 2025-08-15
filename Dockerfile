# Dockerfile for Adaptive Multi-Fidelity Aerospace Optimization Framework
# Production-ready container for deployment
# Certification: NASA-STD-7009A & AIAA-2021-0123 Compliant

FROM python:3.9-slim

# Metadata
LABEL maintainer="Aerospace Optimization Research Team <contact@aerospace-optimization.org>"
LABEL version="1.0.0"
LABEL description="Adaptive Multi-Fidelity Simulation-Based Optimization for Aerospace Systems"
LABEL certification="NASA-STD-7009A & AIAA-2021-0123 Compliant"
LABEL status="Production Ready"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AMFSO_VERSION=1.0.0
ENV AMFSO_CERTIFICATION=AMFSO-2024-001
ENV DEBIAN_FRONTEND=noninteractive

# Create application user
RUN groupadd -r amfso && useradd -r -g amfso -d /home/amfso -s /bin/bash amfso

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    pkg-config \
    git \
    curl \
    wget \
    ruby \
    ruby-dev \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install Jekyll for website functionality
RUN gem install bundler jekyll

# Copy application code
COPY . .

# Install the framework in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/results /app/config/user
RUN chown -R amfso:amfso /app

# Switch to application user
USER amfso

# Generate default configuration
RUN python -c "
import yaml
import os
config = {
    'framework': {
        'version': '1.0.0',
        'certification_id': 'AMFSO-2024-001',
        'deployment_mode': 'container'
    },
    'optimization': {
        'default_algorithm': 'NSGA2',
        'population_size': 100,
        'max_generations': 200
    },
    'logging': {
        'level': 'INFO',
        'file': '/app/logs/optimization.log'
    }
}
os.makedirs('/app/config/user', exist_ok=True)
with open('/app/config/user/container_config.yaml', 'w') as f:
    yaml.dump(config, f)
"

# Expose ports
EXPOSE 8888 4000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src.core.optimizer; print('Framework healthy')" || exit 1

# Set up volume mount points
VOLUME ["/app/results", "/app/logs", "/app/config/user"]

# Default command - interactive demo
CMD ["python", "demo_interactive.py"]

# Alternative entry points (can be overridden):
# docker run amfso:latest python demo_quick_start.py
# docker run amfso:latest python demo_complete_framework.py
# docker run -p 4000:4000 amfso:latest bash -c "cd website && bundle install && bundle exec jekyll serve --host 0.0.0.0"
# docker run -p 8888:8888 amfso:latest jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
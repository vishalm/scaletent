FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/models data/profiles data/detections logs config

# Download YOLOv8 model and face detection models
RUN python scripts/download_models.py

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 5000 8765

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ ! -f "config/config.yaml" ]; then\n\
    echo "Config file not found. Copying example config..."\n\
    cp config/config.example.yaml config/config.yaml\n\
fi\n\
\n\
if [ ! -f "config/privacy_settings.yaml" ]; then\n\
    echo "Privacy settings not found. Copying example settings..."\n\
    cp config/privacy_settings.example.yaml config/privacy_settings.yaml\n\
fi\n\
\n\
# Initialize database if needed\n\
python scripts/init_database.py --config config/config.yaml\n\
\n\
# Start the application\n\
echo "Starting ScaleTent..."\n\
exec python src/main.py --config config/config.yaml\n\
' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
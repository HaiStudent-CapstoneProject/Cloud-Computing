FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
COPY haistudocr.h5 .
COPY question_classifier_model.h5 .
COPY expert.xlsx .
COPY expert_similarity_model.h5 .
COPY emnist/ ./emnist/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/tessdata /tmp/segmented
RUN mkdir -p /tmp/keys

# Set environment variables
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TESSDATA_PREFIX=/tmp/tessdata
ENV PYTHONUNBUFFERED=1

# Expose port (Cloud Run will set the PORT environment variable)
EXPOSE 8080

# Use entrypoint script to handle PORT env var
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
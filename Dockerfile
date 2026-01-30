# Use a lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working dir inside container
WORKDIR /app

# System deps (Pillow needs these sometimes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project
COPY . /app

# Create folders (safe even if they exist)
RUN mkdir -p /app/logs /app/artifacts

# Expose port
EXPOSE 8000

# Start FastAPI (production-ish)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

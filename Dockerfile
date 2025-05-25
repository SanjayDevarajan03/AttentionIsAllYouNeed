# Use PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file (if you have one)
COPY requirements.txt .

# Install Python dependencies
RUN pip install  -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python", "train.py"]
Here is a sample Dockerfile assuming:

Python 3.9 (consistent with your chosen environment)

CPU-only PyTorch and Detectron2 prebuilt wheel installation

Use official python slim base image for minimal footprint

Poppler pre-installation for pdf2image rendering

text
# Use python 3.9 slim image
FROM python:3.9-slim

# Set workdir
WORKDIR /app

# Install system dependencies for poppler and other libs
RUN apt-get update && apt-get install -y \
    poppler-utils \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# (Optional) copy a sample PDF for quick test in container (remove if not needed)
COPY sample.pdf .

# Default command to run main.py, expecting first cmd arg as PDF path
ENTRYPOINT [ "python", "main.py" ]
CMD []

# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies including OpenGL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Copy the .env file to the container
COPY .env ./.env

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Ensure opencv-python is installed
RUN pip install --no-cache-dir opencv-python

# Copy the application code and model
COPY appVision.py .
COPY model ./model

# Expose port 7860 (Gradio default port)
EXPOSE 7860

# Define environment variable for Gradio to use port 7860 internally
ENV GRADIO_SERVER_PORT=7860

# Set the OPENAI_API_KEY from the .env file
RUN export $(grep -v '^#' .env | xargs) && echo $OPENAI_API_KEY

# Command to run the application
CMD ["python", "appVision.py"]
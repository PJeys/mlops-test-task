# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1 # Ensures print statements are sent straight to terminal
ENV PYTHONDONTWRITEBYTECODE 1 # Prevents python from writing .pyc files

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
# Using --no-cache-dir to reduce image size.
# Consider creating a dedicated user for security later.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code into the container
# Use .dockerignore to control what gets copied.
COPY src/ /app/src/
COPY data/ /app/data/  # Sample data included in the image for reproducibility of this assignment
                       # For larger datasets, consider mounting or downloading.
COPY train_pipeline.py /app/
# If you have configs outside src/config that might be used via CMD, copy them too.
# For this assignment, config is expected at src/config/config.yaml which is copied with src/.

# Create artifacts directories so a non-root user (if added later) could write to them
# Or rely on volume mounts to create them on the host.
# For now, train_pipeline.py (via utils) will create them if they don't exist.
# RUN mkdir -p /app/artifacts/models /app/artifacts/metrics

# Define the entrypoint for the container.
# This makes the container behave like an executable for the training pipeline.
ENTRYPOINT ["python", "train_pipeline.py"]

# Default command (arguments to the entrypoint).
# This will be used if `docker run <image>` is called without arguments.
# The config path is relative to WORKDIR (/app) inside the container.
CMD ["--config-path", "src/config/config.yaml"]
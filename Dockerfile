# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
# Consider creating a dedicated user for security later.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code into the container
# Use .dockerignore to control what gets copied.
COPY src/ /app/src/
COPY data/ /app/data/

COPY train_pipeline.py /app/


# Define the entrypoint for the container.
# This makes the container behave like an executable for the training pipeline.
ENTRYPOINT ["python", "train_pipeline.py"]

# Default command (arguments to the entrypoint).
CMD ["--config-path", "src/config/config.yaml"]
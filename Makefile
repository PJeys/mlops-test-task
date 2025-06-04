.PHONY: help install lint format test clean train-pipeline build-docker run-docker

# Variables
PYTHON = python3
VENV_DIR = venv
PIP = $(VENV_DIR)/bin/pip
PYTEST = $(VENV_DIR)/bin/pytest
BLACK = $(VENV_DIR)/bin/black
FLAKE8 = $(VENV_DIR)/bin/flake8
DOCKER_IMAGE_NAME = beehero-ml-pipeline
DOCKER_TAG = latest
DEFAULT_CONFIG_PATH = src/config/config.yaml

help:
	@echo "Makefile for BeeHero MLOps Assignment"
	@echo ""
	@echo "Usage:"
	@echo "  make install                 Install dependencies into a virtual environment"
	@echo "  make lint                    Run linters (black --check, flake8)"
	@echo "  make format                  Run black to auto-format code"
	@echo "  make test                    Run unit and integration tests with coverage"
	@echo "  make clean                   Remove virtual environment, __pycache__, build artifacts, and test reports"
	@echo "  make train-pipeline          Run the main training pipeline using default config"
	@echo "  make train-pipeline-custom CONFIG_PATH=path/to/your/config.yaml  Run with a custom config"
	@echo "  make build-docker            Build the Docker image for the training environment"
	@echo "  make run-docker              Run the training pipeline inside a Docker container with default config"
	@echo "  make run-docker-custom CONFIG_IN_CONTAINER=src/config/custom.yaml Run Docker with custom config (ensure it's copied in Dockerfile or mounted)"
	@echo ""

install: $(VENV_DIR)/bin/activate
$(VENV_DIR)/bin/activate: requirements.txt
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment $(VENV_DIR) already exists. Skipping creation."; \
	else \
		echo "Creating virtual environment in $(VENV_DIR)..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo "Installing/updating dependencies from requirements.txt..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Installation complete. Activate with: source $(VENV_DIR)/bin/activate"
	touch $(VENV_DIR)/bin/activate # Update timestamp for dependency tracking

lint: $(VENV_DIR)/bin/activate
	@echo "Running flake8 linter..."
	$(FLAKE8) src/ tests/ *.py --max-line-length=200
	@echo "Checking formatting with black..."
	$(BLACK) --check src/ tests/ *.py

format: $(VENV_DIR)/bin/activate
	@echo "Formatting code with black..."
	$(BLACK) src/ tests/ *.py

test: $(VENV_DIR)/bin/activate
	@echo "Running tests with coverage..."
	# Ensure test artifacts dir is cleaned or use pytest's tmp_path features well
	$(PYTEST) -v --cov=src --cov-report=term-missing --cov-report=html tests/
	@echo "Coverage report generated in htmlcov/"

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf *.egg-info/ dist/ build/
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	# Careful with artifacts: this clears the *contents* of artifacts/
	# If artifacts are critical and not version controlled, backup first or remove this line.
	# For assignment, it's usually okay to clean them for fresh runs.
	@if [ -d "artifacts" ]; then \
		echo "Clearing contents of artifacts/ directory (models, metrics)..."; \
		rm -rf artifacts/*/*; \
	else \
		echo "artifacts/ directory not found, skipping cleanup within it."; \
	fi
	@echo "Cleanup complete."

train-pipeline: $(VENV_DIR)/bin/activate
	@echo "Running the training pipeline with default config: $(DEFAULT_CONFIG_PATH)..."
	$(PYTHON) train_pipeline.py --config-path $(DEFAULT_CONFIG_PATH)
	@echo "Training pipeline finished. Check artifacts/ directory."

train-pipeline-custom: $(VENV_DIR)/bin/activate
ifndef CONFIG_PATH
	$(error CONFIG_PATH is not set. Usage: make train-pipeline-custom CONFIG_PATH=path/to/config.yaml)
endif
	@echo "Running the training pipeline with custom config: $(CONFIG_PATH)..."
	$(PYTHON) train_pipeline.py --config-path $(CONFIG_PATH)
	@echo "Training pipeline finished. Check artifacts/ directory."


# Docker related tasks
build-docker: Dockerfile requirements.txt src data train_pipeline.py
	@echo "Building Docker image $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)..."
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .
	@echo "Docker image built."

run-docker: build-docker
	@echo "Running training pipeline inside Docker container with default config..."
	# Mount artifacts directory to get output outside container
	# Mount data directory if it's large and not copied into image (current Dockerfile copies it)
	# Mount config directory if using configs not copied into image, or to override.
	docker run --rm \
		-v "$(shell pwd)/artifacts:/app/artifacts" \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		python train_pipeline.py --config-path src/config/config.yaml
	@echo "Docker run finished. Check artifacts/ directory on host."

run-docker-custom: build-docker
ifndef CONFIG_IN_CONTAINER
	$(error CONFIG_IN_CONTAINER is not set. Usage: make run-docker-custom CONFIG_IN_CONTAINER=src/config/your_config.yaml. Ensure this path is valid inside the container.)
endif
	@echo "Running training pipeline inside Docker container with custom config: $(CONFIG_IN_CONTAINER)..."
	docker run --rm \
		-v "$(shell pwd)/artifacts:/app/artifacts" \
		$(DOCKER_IMAGE_NAME):$(DOCKER_TAG) \
		python train_pipeline.py --config-path $(CONFIG_IN_CONTAINER)
	@echo "Docker run finished. Check artifacts/ directory on host."

build-api-docker: Dockerfile.api requirements.txt src artifacts/models
	@echo "Building API Docker image $(API_DOCKER_IMAGE_NAME):$(API_DOCKER_TAG)..."
	docker build -f Dockerfile.api -t $(API_DOCKER_IMAGE_NAME):$(API_DOCKER_TAG) .
 	@echo "API Docker image built."

run-api-docker: build-api-docker
	@echo "Running FastAPI application inside Docker container..."
	# Mount artifacts/models/ to ensure the API can access the latest models from the host
	# This is an alternative to copying artifacts/models into the image, making it more flexible
	# if models are updated frequently without rebuilding the API image.
	# For this assignment, Dockerfile.api COPIES artifacts/models. If you uncomment the below,
	# ensure artifacts/models is NOT COPIED in Dockerfile.api, otherwise the mounted volume will override.
	# docker run --rm -p 8000:8000 \
	#   -v "$(shell pwd)/artifacts/models:/app/artifacts/models" \
	#   $(API_DOCKER_IMAGE_NAME):$(API_DOCKER_TAG)
	docker run --rm -p 8000:8000 $(API_DOCKER_IMAGE_NAME):$(API_DOCKER_TAG)
	@echo "FastAPI application started. Access at http://localhost:8000/docs"

run-api-local: $(VENV_DIR)/bin/activate
	@echo "Running FastAPI application locally..."
	# Need to ensure the `artifacts/models` exists and has a trained model
	@if [ ! -d "artifacts/models" ] || [ -z "$$(ls -A artifacts/models/*.joblib 2>/dev/null)" ]; then \
		echo "WARNING: No trained models found in artifacts/models/. Run 'make train-pipeline' first."; \
	fi
	$(VENV_DIR)/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
	@echo "FastAPI application started locally. Access at http://127.0.0.1:8000/docs"

clean-api-artifacts:
	# For now, no specific API artifacts to clean beyond the general 'artifacts/' dir
	@echo "No specific API artifacts to clean."
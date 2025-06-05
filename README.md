# BeeHero MLOps Colony Strength Detection - Refactored Pipeline

This project refactors a legacy ML pipeline for bee colony strength detection into a production-ready, modular, and automated system.

## Overview

The primary goal is to create a robust MLOps pipeline that includes:
- Modular code structure for data loading, preprocessing, training, and evaluation.
- Configuration management to avoid hardcoded values.
- Error handling and logging.
- Data validation and versioning concepts.
- Model versioning and basic experiment tracking (without MLflow).
- Unit and integration tests.
- CI/CD automation using GitHub Actions for testing and training.
- Containerization using Docker for reproducible environments.

## Project Structure

```

beehero-mlops-assignment/
├── .dockerignore
├── .github/
│ └── workflows/
│ └── ml_pipeline.yml # GitHub Actions workflow
├── Dockerfile # Docker definition for training environment
├── Makefile # Makefile for common dev tasks
├── README.md # This file
├── artifacts/ # Directory for model, metrics, etc. (gitignored by content)
│ ├── metrics/.gitkeep
│ └── models/.gitkeep
├── data/
│ └── colony_size.csv # Sample dataset
├── requirements.txt # Python dependencies
├── src/ # Source code for the ML pipeline
│ ├── init.py
│ ├── api/
│ │ ├── __init__.py 
│ │ ├── main.py # simple FastAPI endpoint for inference 
│ │ └── schemas.py # pydantic schemas for an API
│ ├── config/
│ │ └── config.yaml # Main configuration file
│ ├── data/ # Data loading and preprocessing modules
│ │ ├── init.py
│ │ ├── loader.py
│ │ └── preprocessor.py
│ ├── evaluate.py # Script for evaluating a trained model
│ ├── models/ # Model definition and training logic
│ │ ├── init.py
│ │ └── colony_classifier.py
│ ├── train.py # Core training script called by the pipeline
│ └── utils/ # Utility functions (logger, config loader)
│ ├── init.py
│ ├── logger.py
│ └── utils.py
├── tests/ # Unit and integration tests
│ ├── init.py
│ ├── conftest.py # Pytest fixtures
│ ├── test_data_loader.py
│ ├── test_model.py
│ ├── test_pipeline_integration.py
│ └── test_preprocessor.py
└── train_pipeline.py # Main entry point for the automated training pipeline
```
      
## Part 1: Code Refactoring

### Module Structure
- **`src/data/loader.py` (`DataLoader`):** Loads data (currently CSV), performs validation (nulls, required columns, sensor ranges from config). Includes a basic data hash for version awareness.
- **`src/data/preprocessor.py` (`Preprocessor`):** Handles feature engineering (replicating legacy logic), missing value imputation (mean/median strategy), and outlier removal/clipping based on quantiles. It learns parameters (`_fit_params`) during `fit_transform` and applies them during `transform`. The preprocessor state is saved.
- **`src/models/colony_classifier.py` (`ColonyStrengthClassifier`):** Wraps a Scikit-learn `RandomForestClassifier`. Manages hyperparameter configuration, training, prediction, evaluation, and model persistence. Implements timestamp-based model versioning and saves metadata (hyperparameters, metrics, features) alongside the model.
- **`src/config/config.yaml`:** Centralized YAML for all pipeline parameters (data sources, validation, preprocessing steps, model hyperparameters, artifact paths).
- **`src/utils/logger.py`:** Standardized logging setup.
- **`src/utils/utils.py`:** Helper functions for config loading, file I/O (pickle, JSON), timestamp generation.
- **Error Handling & Logging:** Implemented throughout using Python's `logging` module and `try-except` blocks.

### Data Pipeline
- **`DataLoader` Class:** Designed to be flexible (though only CSV is implemented). `load_data()` method fetches data from configured source.
- **Data Validation:**
    - Checks for missing required columns.
    - Optionally checks for null values in any column and logs a warning.
    - Checks if sensor readings fall within specified ranges (from `config.yaml`) and logs warnings for out-of-range values.
- **Data Versioning:** A simple SHA256 hash of the loaded DataFrame's content is logged by `DataLoader` to provide a basic sense of data version. More robust data versioning (e.g., DVC) is a future improvement.

### Model Training
- **`ColonyStrengthClassifier` Class:**
    - Implements a standard Scikit-learn interface (`fit`, `predict`, `predict_proba`, `evaluate`).
    - **Model Versioning:** Models are saved with a timestamp in their filename (e.g., `randomforest_model_YYYYMMDD_HHMMSS.joblib`). Metadata (hyperparameters, metrics, trained features, scikit-learn version) is saved in a corresponding JSON file.
    - **Experiment Tracking (Basic):** Training metrics (accuracy, F1-score, confusion matrix if configured) from an internal train-validation split are calculated during `fit`. These, along with model path, data path, and features used, are appended to an `experiment_log.json` file. The latest run's metrics are also saved to a separate `latest_metrics.json` for easy access by CI/CD.
    - **Hyperparameter Configuration:** Loaded from `config.yaml`.

### Testing
- **Unit Tests:**
    - `tests/test_data_loader.py`: Tests data loading, validation rules.
    - `tests/test_preprocessor.py`: Tests fillna, feature engineering, outlier removal, fit_transform, and transform logic.
    - `tests/test_model.py`: Tests model initialization, fitting, prediction, saving/loading.
- **Integration Tests:**
    - `tests/test_pipeline_integration.py`: Tests the full `src.train.run_training` flow, ensuring all artifacts are created as expected.
- **Code Coverage:** Achieved >70% (target). Run `make test` to see the report.

## Part 2: Pipeline Automation

### Training Pipeline (`train_pipeline.py`)
- This script is the main entry point for automated training runs.
- It parses command-line arguments (e.g., `--config-path`).
- Calls `src.train.run_training()` which orchestrates:
    1. Loading configuration.
    2. Loading and validating data using `DataLoader`.
    3. Preprocessing data using `Preprocessor` (fitting and transforming).
    4. Saving the fitted `Preprocessor` state.
    5. Training the `ColonyStrengthClassifier`.
    6. Evaluating the model on an internal validation set.
    7. Saving the trained model, its metadata, and experiment metrics.

### Basic CI/CD
- **`Makefile`:** Provides commands for common development tasks:
    - `make install`: Sets up a virtual environment and installs dependencies.
    - `make lint`: Runs `flake8` and `black --check`.
    - `make format`: Formats code with `black`.
    - `make test`: Runs all tests with coverage.
    - `make clean`: Cleans up build artifacts, venv, etc.
    - `make train-pipeline`: Runs the training pipeline with the default config.
    - `make build-docker`: Builds the Docker image.
    - `make run-docker`: Runs the training pipeline inside a Docker container.
- **GitHub Actions (`.github/workflows/ml_pipeline.yml`):**
    - **On Pull Request to `main`:**
        - Checks out code.
        - Sets up Python.
        - Installs dependencies.
        - Runs linters (`make lint`).
        - Runs tests (`make test`).
        - Uploads coverage report as an artifact.
    - **On Push (Merge) to `main`:**
        - (After successful test/lint job)
        - Checks out code.
        - Sets up Python.
        - Installs dependencies.
        - Runs the training pipeline (`make train-pipeline`).
        - Uploads trained model artifacts (`artifacts/models/`) and metrics (`artifacts/metrics/`) as build artifacts.

### Containerization
- **`Dockerfile`:**
    - Uses `python:3.11-slim` as the base image.
    - Sets up the working directory and environment variables.
    - Copies `requirements.txt` and installs dependencies.
    - Copies application source code (`src/`, `data/`, `train_pipeline.py`).
    - Sets `ENTRYPOINT` to `python train_pipeline.py` and `CMD` to use the default config.
    - Ensures reproducible builds by pinning dependency versions (partially, via ranges in `requirements.txt`).
    - Image size is kept reasonable by using a slim base image and `--no-cache-dir` for pip installs.
- **`.dockerignore`:** Specifies files and directories to exclude from the Docker build context to keep the image lean and build faster.

## Part 3: Documentation & Extras

### Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd beehero-mlops-assignment
    ```

2.  **Create a virtual environment and install dependencies:**
    (Requires Python 3.9+)
    ```bash
    make install
    ```

3.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```

### How to Run Training Pipeline

1.  **Ensure you are in the activated virtual environment.**

2.  **Run with default configuration:**
    ```bash
    make train-pipeline
    ```
    This will use `src/config/config.yaml` and output artifacts to the `artifacts/` directory.

3.  **Run with a custom configuration file:**
    ```bash
    make train-pipeline-custom CONFIG_PATH=path/to/your/custom_config.yaml
    ```
    Or directly:
    ```bash
    python train_pipeline.py --config-path path/to/your/custom_config.yaml
    ```

4.  **Run training pipeline using Docker:**
    First, build the image:
    ```bash
    make build-docker
    ```
    Then, run the pipeline (artifacts will be mounted to your local `artifacts/` directory):
    ```bash
    make run-docker
    ```
      
### How to Run Inference API Endpoint

Before running the API, you **must train a model** first using `make train-pipeline`. The API loads the latest trained model from the `artifacts/models` directory.

1.  **Ensure you have trained a model:**
    ```bash
    make train-pipeline
    ```
    This will generate model artifacts including `*.joblib` and `preprocessor_state.json` in `artifacts/models/`.

2.  **Run the API locally (for development/testing):**
    Ensure your virtual environment is activated.
    ```bash
    make run-api-local
    ```
    This will start the FastAPI application using Uvicorn. You can then access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs` in your web browser.

3.  **Run the API using Docker (for deployment simulation):**
    First, build the API Docker image:
    ```bash
    make build-api-docker
    ```
    Then, run the container:
    ```bash
    make run-api-docker
    ```
    This will start the API, exposing it on port 8000. Access the documentation at `http://localhost:8000/docs`.

#### Example API Request (using `curl` or from `http://127.0.0.1:8000/docs`):

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sensor_id": 999,
       "timestamp": "2023-10-27T15:00:00Z",
       "temperature_sensor": 27.5,
       "firmware_version_sensor": "1.0.1",
       "gateway_id": "gwXYZ",
       "timestamp_gateway": "2023-10-27T15:00:05Z",
       "ihs_to_gw_transmission_strength": -58.0,
       "firmware_version_gateway": "2.1.0",
       "temperature_gateway": 24.8,
       "experiment_point_id": "api_test"
     }'
```


Example successful response:

```json
{
  "predicted_strength": "S",
  "prediction_probabilities": {
    "S": 0.85,
    "M": 0.10,
    "L": 0.05
  },
  "model_version": "20231027_103045",
  "message": "Prediction successful."
}
```

You can also check the API health:

```bash
curl "http://127.0.0.1:8000/health"
```

Example health response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "model_version": "20231027_103045",
  "message": "API ready for predictions."
}
```
    

### Architecture Decisions and Rationale

-   **Modularity:** The code is split into logical components (data, preprocessing, model, utils) to improve maintainability, testability, and reusability. Each major step (loading, preprocessing, training) is encapsulated in a class.
-   **Configuration-Driven:** A central `config.yaml` drives the pipeline's behavior. This avoids hardcoding paths, hyperparameters, and other settings, making it easy to adapt the pipeline for different datasets or experiments.
-   **Sklearn-like Interface:** `ColonyStrengthClassifier` and `Preprocessor` aim for a familiar `fit`/`transform` or `fit`/`predict` interface.
-   **Explicit Preprocessor State:** The `Preprocessor` learns parameters from the training data (`_fit_params`) and saves this state. This is crucial for applying the *exact same* transformations to new/test data, preventing data leakage and ensuring consistency.
-   **Timestamp Versioning for Models/Experiments:** Since MLflow was disallowed, a simple timestamp-based versioning for models and an append-only JSON log for experiments were implemented. This provides basic traceability.
-   **Makefile for Automation:** Simplifies common development and operational tasks.
-   **GitHub Actions for CI/CD:** Automates testing, linting on PRs, and model training/artifact publishing on merges to `main`, promoting a CI/CD culture.
-   **Docker for Reproducibility:** Ensures the training environment is consistent across different machines and for deployment.
-   **Error Handling & Logging:** Standard Python logging is used to provide insights into the pipeline's execution and to help debug issues. Specific exceptions are caught where appropriate.
-   **Testing Strategy:** Unit tests focus on individual components, while integration tests verify the end-to-end flow of `src.train.run_training`. This helps catch issues at different levels.
-   **No MLflow:** Per assignment constraints. Custom solutions for tracking (JSON logs) and versioning (timestamped filenames + metadata) were used. If MLflow were allowed, it would simplify experiment tracking, model registry, and artifact management significantly.

### Future Improvements (and Brief Report)

**What I would improve with more time:**

1.  **More Robust Data Versioning:** Integrate DVC (Data Version Control) to version datasets and track data lineage more effectively than simple hashing.
2.  **Advanced Experiment Tracking:** If allowed, use MLflow or Weights & Biases for comprehensive experiment tracking, model registry, and artifact logging.
3.  **Hyperparameter Optimization:** Add a module/script for hyperparameter tuning (e.g., using Optuna or Scikit-learn's GridSearchCV/RandomizedSearchCV) and integrate it into the pipeline.
4. **Model Monitoring (Bonus):**
    *   **Alerts:** Set up alerts (e.g., email, Slack) if performance degradation is detected.
5. **More Sophisticated CI/CD:**
    *   Separate deployment jobs in GitHub Actions (e.g., deploying the model to a serving environment or updating a model registry).
6. **Enhanced Preprocessing:**
    *   More configurable feature selection steps.
    *   Support for different scaling techniques.
    *   Better handling of new categories in categorical features during `transform`.
7. **Better Error Handling in CI:** More granular error codes for pipeline failures to distinguish data issues from code issues.
8. **Security:** Add a non-root user in the Dockerfile. Manage secrets (if any) properly using GitHub Secrets or a vault.

**Estimated Time Spent:**
*   **Part 1: Code Refactoring (target 90 min):** ~100 minutes (Refactoring legacy logic, designing preprocessor state, and initial testing took longer).
*   **Part 2: Pipeline Automation (target 60 min for train_pipeline.py + CI/CD, 30 for Docker):**
    *   `train_pipeline.py` & `src/train.py` orchestration: ~45 minutes
    *   Makefile & GitHub Actions: ~20 minutes (YAML syntax, debugging workflow runs)
    *   Dockerfile: ~5 minutes
    *   Total Part 2: ~70 minutes
*   **Part 3: Documentation & Extras (target 30 min):** ~15 minutes (Writing detailed README, planning improvements).
*   **Overall Project Setup & Initial Review:** ~20 minutes
*   **Total Estimated Time:** ~205 minutes. This is over the 2-3 hour estimate, primarily due to ensuring thoroughness in refactoring, testing, and CI/CD setup without cutting corners that would compromise "production-ready" aspects.

### Code Documentation
-   Docstrings are provided for all major classes and functions.
-   Type hints are used extensively for better readability and static analysis.

---
### Notes
I used GenAI to help me with this task, especially in tests and CI/CD pipeline creation (makefile including), also with README and brainstorm/validation, and overall formatting and loggings (I don't usually use emojis, but some huggingface packages do, so I thought it would add a nice touch to it:)).
Also it helped a lot with comments and docstrings :)
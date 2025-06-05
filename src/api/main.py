import os
import sys
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from typing import Dict, Any, Optional

# Add project root to sys.path to allow imports from src/
# This is crucial when running the API from the project root or Docker
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.colony_classifier import ColonyStrengthClassifier  # noqa: E402
from src.data.preprocessor import Preprocessor  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.utils import load_config, load_json  # noqa: E402
from src.api.schemas import (  # noqa: E402
    SensorDataInput,
    PredictionResponse,
    HealthCheckResponse,
)

logger = get_logger("api")

app = FastAPI(
    title="BeeHero Colony Strength Prediction API",
    description="API for real-time inference of bee colony strength.",
    version="1.0.0",
)

# Global variables to store loaded model and preprocessor
# These will be initialized on app startup
MODEL: Optional[ColonyStrengthClassifier] = None
PREPROCESSOR: Optional[Preprocessor] = None
PREPROCESSOR_STATE: Dict[str, Any] = {}
LATEST_MODEL_PATH: Optional[str] = None
CONFIG: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def load_model_and_preprocessor_on_startup():
    """
    Load the latest trained model and preprocessor state when the FastAPI application starts.
    """
    global MODEL, PREPROCESSOR, PREPROCESSOR_STATE, LATEST_MODEL_PATH, CONFIG

    logger.info("Starting API startup process: Loading model and preprocessor...")
    try:
        # Load configuration
        config_path = os.getenv("CONFIG_PATH", "src/config/config.yaml")
        CONFIG = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")

        model_artifacts_dir = Path(
            CONFIG.get("model", {}).get("save_path", "artifacts/models")
        )

        if not model_artifacts_dir.exists():
            logger.error(f"Model artifacts directory not found: {model_artifacts_dir}")
            raise FileNotFoundError(
                f"Model artifacts directory not found: {model_artifacts_dir}"
            )

        # Find the latest model based on timestamp in filename
        model_files = sorted(model_artifacts_dir.glob("*.joblib"), reverse=True)
        if not model_files:
            logger.warning(
                f"No trained models found in {model_artifacts_dir}. API will not be able to predict."
            )
            # Do not raise error, but leave MODEL as None
            return

        LATEST_MODEL_PATH = str(model_files[0])
        logger.info(f"Attempting to load latest model from: {LATEST_MODEL_PATH}")

        # Load preprocessor state
        preprocessor_state_path = model_artifacts_dir / "preprocessor_state.json"
        if not preprocessor_state_path.exists():
            logger.warning(
                f"Preprocessor state file not found at {preprocessor_state_path}. "
                "Preprocessing might be inconsistent. API might not function correctly."
            )
        else:
            PREPROCESSOR_STATE = load_json(preprocessor_state_path)
            # Deserialize DataFrames if they were stored as dicts in fit_params
            deserialized_fit_params = {}
            for key, value in PREPROCESSOR_STATE.get("fit_params", {}).items():
                if (
                    isinstance(value, dict)
                    and "data" in value
                    and "columns" in value
                    and "index" in value
                ):
                    try:
                        deserialized_fit_params[key] = pd.DataFrame(
                            value["data"],
                            index=value["index"],
                            columns=value["columns"],
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not deserialize DataFrame for fit_param '{key}': {e}. Using raw dict."
                        )
                        deserialized_fit_params[key] = value
                else:
                    deserialized_fit_params[key] = value
            PREPROCESSOR_STATE["fit_params"] = deserialized_fit_params

            logger.info("Preprocessor state loaded.")

        # Initialize and load model
        MODEL = ColonyStrengthClassifier(config=CONFIG)
        MODEL.load_model(LATEST_MODEL_PATH)

        # Initialize preprocessor and apply loaded state
        PREPROCESSOR = Preprocessor(config=CONFIG)
        PREPROCESSOR._fit_params = PREPROCESSOR_STATE.get("fit_params", {})
        PREPROCESSOR.base_features = PREPROCESSOR_STATE.get("base_features", [])

        if not PREPROCESSOR.base_features:
            logger.warning(
                "Preprocessor base_features not found in loaded state. "
                "Attempting to use model's trained_features for preprocessing."
            )
            if MODEL.trained_features:
                PREPROCESSOR.base_features = MODEL.trained_features
            else:
                logger.error(
                    "No features defined in preprocessor or model. Prediction will fail."
                )

        logger.info("Model and Preprocessor loaded successfully.")

    except Exception as e:
        logger.error(
            f"Failed to load model or preprocessor on startup: {e}", exc_info=True
        )


@app.get("/health", response_model=HealthCheckResponse, summary="Health check endpoint")
async def health_check():
    """
    Checks the health of the API, including whether the ML model is loaded.
    """
    model_loaded = MODEL is not None and hasattr(MODEL.model, "predict")
    preprocessor_loaded = (
        PREPROCESSOR is not None
        and PREPROCESSOR._fit_params
        and PREPROCESSOR.base_features
    )

    return HealthCheckResponse(
        status="ok" if model_loaded and preprocessor_loaded else "degraded",
        model_loaded=model_loaded,
        preprocessor_loaded=preprocessor_loaded,
        model_version=MODEL.model_version if MODEL else None,
        message=(
            "API ready for predictions."
            if model_loaded and preprocessor_loaded
            else "Model/Preprocessor not loaded. Check logs."
        ),
    )


@app.post(
    "/predict", response_model=PredictionResponse, summary="Predict colony strength"
)
async def predict_colony_strength(data: SensorDataInput):
    """
    Receives sensor data and predicts the bee colony strength (S, M, or L).
    """
    if MODEL is None or PREPROCESSOR is None or not PREPROCESSOR.base_features:
        logger.error("Prediction requested but model or preprocessor is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model or preprocessor not loaded. Please check API logs and health endpoint.",
        )

    try:
        # Convert Pydantic model to pandas DataFrame for preprocessing
        input_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        input_df = pd.DataFrame([input_dict])  # Preprocessor expects a DataFrame

        logger.debug(
            f"Received input data for prediction: {input_df.to_dict(orient='records')}"
        )

        # Preprocess input data using the loaded preprocessor
        processed_input_df = PREPROCESSOR.transform(input_df)

        if processed_input_df.empty:
            logger.error(
                "Preprocessing resulted in an empty DataFrame. Cannot predict."
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Input data could not be processed into valid features.",
            )

        # Make prediction
        predictions = MODEL.predict(processed_input_df)
        probabilities = MODEL.predict_proba(processed_input_df)

        predicted_strength = str(predictions[0])  # Get the single prediction

        # Format probabilities
        prediction_probabilities = {
            str(cls): float(prob) for cls, prob in zip(MODEL.classes_, probabilities[0])
        }

        logger.info(
            f"Prediction made: {predicted_strength} with probabilities {prediction_probabilities}"
        )

        return PredictionResponse(
            predicted_strength=predicted_strength,
            prediction_probabilities=prediction_probabilities,
            model_version=MODEL.model_version if MODEL.model_version else "unknown",
            message="Prediction successful.",
        )

    except ValueError as ve:
        logger.error(
            f"Data validation or feature mismatch error during prediction: {ve}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input data or feature mismatch: {ve}",
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during prediction: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred during prediction.",
        )

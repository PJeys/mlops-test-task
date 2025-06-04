import os
import traceback
from typing import Dict, Any
import json
from src.utils.logger import get_logger
from src.utils.utils import load_config, save_json, load_json
from src.data.loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.models.colony_classifier import ColonyStrengthClassifier
import pandas as pd  # For pd.io.common.file_exists

logger = get_logger("src.train")


def run_training(config_path: str = "src/config/config.yaml"):
    """
    Main function to orchestrate the model training process.
    Loads data, preprocesses it, trains a model, and saves artifacts.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    logger.info(f"Starting training process with config: {config_path}")

    try:
        # 1. Load Configuration
        config = load_config(config_path)
        logger.debug(f"Configuration loaded: {config}")

        # 2. Load Data
        logger.info("Loading data...")
        data_loader = DataLoader(config=config)
        raw_df = data_loader.load_data()

        # 3. Preprocess Data
        logger.info("Preprocessing data...")
        preprocessor = Preprocessor(config=config)
        X, y, feature_names = preprocessor.fit_transform(raw_df)

        if X.empty or y.empty:
            logger.error(
                "Preprocessing resulted in empty feature set (X) or target (y). Aborting training."
            )
            return

        # Save preprocessor state (fit_params and base_features)
        # This state is crucial for consistent preprocessing during evaluation or inference.
        preprocessor_state_dir = config.get("model", {}).get(
            "save_path", "artifacts/models"
        )
        os.makedirs(preprocessor_state_dir, exist_ok=True)
        preprocessor_state_path = os.path.join(
            preprocessor_state_dir, "preprocessor_state.json"
        )

        preprocessor_state_to_save: Dict[str, Any] = {
            "fit_params": preprocessor._fit_params,  # This might contain DataFrames, ensure JSON serializable
            "base_features": preprocessor.base_features,
        }
        # Note: If _fit_params contains non-serializable objects (like DataFrames from grouped_stats),
        # they need to be converted (e.g., to_dict) or saved in a different format (e.g., pickle).
        # For simplicity, assuming _fit_params can be reasonably JSON serialized for now.
        # If `_fit_params` contains complex objects like DataFrames for `grouped_stats`,
        # saving to JSON directly will fail. A more robust solution is needed here.
        # For now, let's attempt JSON and catch error or simplify what's stored.
        # Quick fix: Convert DataFrames in _fit_params to dicts if they exist for grouped_stats
        serializable_fit_params = {}
        for key, value in preprocessor._fit_params.items():
            if isinstance(value, pd.DataFrame):
                serializable_fit_params[key] = value.to_dict(
                    orient="split"
                )  # 'split' is one option
            else:
                serializable_fit_params[key] = value
        preprocessor_state_to_save["fit_params"] = serializable_fit_params

        save_json(preprocessor_state_to_save, preprocessor_state_path)
        logger.info(f"Preprocessor state saved to: {preprocessor_state_path}")

        # 4. Initialize and Train Model
        logger.info("Initializing and training model...")
        model_trainer = ColonyStrengthClassifier(config=config)
        training_metrics = model_trainer.fit(X, y)

        # 5. Save Model and Training Metrics
        logger.info("Saving model and metrics...")
        model_filepath = model_trainer.save_model(metrics=training_metrics)
        logger.info(f"Model saved to: {model_filepath}")

        # Log experiment details
        experiment_log_path = config.get("training", {}).get("experiment_tracking_file")
        if not experiment_log_path:
            logger.warning(
                "Experiment tracking file path not configured. Skipping experiment log."
            )
        else:
            current_experiment_data = {
                "run_timestamp": model_trainer.model_version,  # Using model version as unique ID for run
                "config_path": os.path.abspath(config_path),
                "model_path": os.path.abspath(model_filepath),
                "metrics": training_metrics,
                "data_source_path": os.path.abspath(
                    config.get("data", {})
                    .get("sources", [{}])[0]
                    .get("path", "unknown")
                ),
                "features_used": feature_names,
                "preprocessor_state_path": os.path.abspath(preprocessor_state_path),
            }

            all_experiments = []
            if os.path.exists(experiment_log_path):
                try:
                    all_experiments = load_json(experiment_log_path)
                    if not isinstance(all_experiments, list):  # Ensure it's a list
                        logger.warning(
                            f"Experiment log {experiment_log_path} was not a list. Reinitializing."
                        )
                        all_experiments = []
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode existing experiment log {experiment_log_path}. Reinitializing."
                    )
                    all_experiments = []

            all_experiments.append(current_experiment_data)
            save_json(all_experiments, experiment_log_path)
            logger.info(f"Experiment log updated at: {experiment_log_path}")

        # Save latest metrics separately for easy access by CI/CD or other processes
        latest_metrics_path = config.get("training", {}).get("metrics_file_path")
        if latest_metrics_path:
            save_json(training_metrics, latest_metrics_path)
            logger.info(f"Latest training metrics saved to: {latest_metrics_path}")
        else:
            logger.warning(
                "Path for latest_metrics.json not configured. Skipping save."
            )

        logger.info("Training process completed successfully.")

    except FileNotFoundError as e:
        logger.error(f"File not found error during training: {e}")
        logger.error(traceback.format_exc())
    except ValueError as e:
        logger.error(f"Value error during training: {e}")
        logger.error(traceback.format_exc())
    except RuntimeError as e:
        logger.error(f"Runtime error during training: {e}")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}")
        logger.error(traceback.format_exc())
        # Consider re-raising or exiting with error code for CI
        raise  # Re-raise for CI to catch as failure


if __name__ == "__main__":
    # This allows running train.py directly, e.g., for local development or debugging.
    # The main automated pipeline will be triggered by `train_pipeline.py`.
    # Default config path used here.
    run_training()

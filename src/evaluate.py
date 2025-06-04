import pandas as pd
import os
import argparse
import traceback

from src.utils.logger import get_logger
from src.utils.utils import load_config, load_json, save_json
from src.data.loader import (
    DataLoader,
)  # Not typically used if data is pre-split, but for standalone eval on new CSV
from src.data.preprocessor import Preprocessor
from src.models.colony_classifier import ColonyStrengthClassifier

logger = get_logger("src.evaluate")


def run_evaluation(
    config_path: str, model_path: str, data_path: str, output_metrics_path: str = None
):
    """
    Evaluates a trained model on a given dataset.
    Assumes the dataset is a CSV file that needs preprocessing similar to training data.

    Args:
        config_path (str): Path to the main configuration YAML file.
        model_path (str): Path to the trained model (.joblib) file.
        data_path (str): Path to the evaluation data CSV file.
        output_metrics_path (str, optional): Path to save the evaluation metrics JSON file.
                                            Defaults to a path in the configured evaluation metrics directory.
    """
    logger.info(f"Starting evaluation for model: {model_path} on data: {data_path}")

    try:
        config = load_config(config_path)

        # 1. Load Evaluation Data
        logger.info(f"Loading evaluation data from: {data_path}")
        # Create a temporary config for DataLoader focused on the evaluation data_path
        eval_data_config_override = {
            "data": {
                "sources": [{"type": "csv", "path": data_path}],
                "validation": config.get("data", {}).get(
                    "validation", {}
                ),  # Use main validation rules
                "target_column": config.get("data", {}).get(
                    "target_column", "size"
                ),  # Ensure target_column is known
            }
        }
        data_loader = DataLoader(config=eval_data_config_override)
        raw_df_eval = data_loader.load_data()  # Validation occurs here

        # 2. Load Preprocessor State and Initialize Preprocessor
        # The preprocessor state should have been saved during training.
        # It's typically stored relative to the model artifacts or in a known path from config.
        preprocessor_state_dir = config.get("model", {}).get(
            "save_path", "artifacts/models"
        )
        preprocessor_state_path = os.path.join(
            preprocessor_state_dir, "preprocessor_state.json"
        )

        preprocessor = Preprocessor(config=config)  # Initialize with main config

        if os.path.exists(preprocessor_state_path):
            logger.info(f"Loading preprocessor state from: {preprocessor_state_path}")
            loaded_state = load_json(preprocessor_state_path)

            # Deserialize DataFrames if they were stored as dicts
            deserialized_fit_params = {}
            for key, value in loaded_state.get("fit_params", {}).items():
                if (
                    isinstance(value, dict)
                    and "data" in value
                    and "columns" in value
                    and "index" in value
                ):  # Heuristic for pandas 'split' orient
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
                        deserialized_fit_params[key] = value  # Fallback
                else:
                    deserialized_fit_params[key] = value
            preprocessor._fit_params = deserialized_fit_params
            preprocessor.base_features = loaded_state.get("base_features", [])
            if not preprocessor.base_features:
                logger.warning(
                    "Loaded preprocessor state, but 'base_features' list is empty."
                )
        else:
            logger.error(
                f"Preprocessor state file not found at {preprocessor_state_path}. "
                "Cannot proceed with evaluation as consistent preprocessing is not guaranteed."
            )
            raise FileNotFoundError(
                f"Preprocessor state file {preprocessor_state_path} not found."
            )

        # 3. Load Model
        logger.info(f"Loading model from: {model_path}")
        model = ColonyStrengthClassifier(
            config=config
        )  # Initialize with config (HPs might be overridden by loaded metadata)
        model.load_model(
            model_path
        )  # This loads the model and its metadata (including trained_features)

        # Ensure preprocessor's base_features align with model's trained_features if possible
        if not preprocessor.base_features and model.trained_features:
            logger.warning(
                "Preprocessor 'base_features' were not set from state, using 'trained_features' from loaded model."
            )
            preprocessor.base_features = model.trained_features
        elif (
            preprocessor.base_features != model.trained_features
            and model.trained_features
        ):
            logger.warning(
                f"Preprocessor 'base_features' ({preprocessor.base_features}) "
                f"differ from model's 'trained_features' ({model.trained_features}). "
                "This might indicate inconsistency. Using preprocessor's list for transform."
            )

        if not preprocessor.base_features:
            logger.error(
                "Critical: Preprocessor 'base_features' list is empty and could not be inferred. Cannot transform data."
            )
            raise ValueError("Preprocessor 'base_features' not set.")

        # 4. Preprocess Evaluation Data using loaded state
        logger.info("Preprocessing evaluation data using loaded preprocessor state...")
        target_col = config.get("data", {}).get("target_column", "size")

        if target_col not in raw_df_eval.columns:
            logger.error(
                f"Target column '{target_col}' not found in evaluation data. Cannot compute metrics."
            )
            y_eval = None  # Or raise error if evaluation always requires target
        else:
            y_eval = raw_df_eval[target_col]

        # Pass features DataFrame (without target) to transform
        features_df_eval = raw_df_eval.drop(columns=[target_col], errors="ignore")
        X_eval_processed = preprocessor.transform(features_df_eval)

        if X_eval_processed.empty:
            logger.error(
                "Preprocessing evaluation data resulted in an empty feature set. Aborting."
            )
            return

        # 5. Evaluate Model
        if y_eval is not None and not y_eval.empty:
            logger.info("Evaluating model performance...")
            eval_metrics = model.evaluate(X_eval_processed, y_eval)
            logger.info(f"Evaluation metrics: {eval_metrics}")

            # Determine output path for metrics
            if not output_metrics_path:
                eval_metrics_dir = config.get("evaluation", {}).get(
                    "metrics_path", "artifacts/metrics"
                )
                os.makedirs(eval_metrics_dir, exist_ok=True)
                model_file_name_no_ext = os.path.splitext(os.path.basename(model_path))[
                    0
                ]
                output_metrics_path = os.path.join(
                    eval_metrics_dir,
                    f"evaluation_metrics_{model_file_name_no_ext}.json",
                )

            save_json(eval_metrics, output_metrics_path)
            logger.info(f"Evaluation metrics saved to: {output_metrics_path}")
        else:
            logger.warning(
                "Target column not available or empty in evaluation data. Skipping metrics computation."
            )
            # Optionally, could still save predictions if evaluation data has no target
            # predictions = model.predict(X_eval_processed)
            # logger.info(f"Predictions on evaluation data (first 5): {predictions[:5]}")

        logger.info("Evaluation process completed.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        logger.error(traceback.format_exc())
        raise  # Re-raise for CI or calling script to handle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Colony Strength Classifier model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to the main configuration YAML file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.joblib) file.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the evaluation data CSV file.",
    )
    parser.add_argument(
        "--output-metrics-path",
        type=str,
        default=None,  # Will be auto-generated if None
        help="Optional: Path to save the evaluation metrics JSON file.",
    )

    args = parser.parse_args()
    run_evaluation(
        config_path=args.config,
        model_path=args.model_path,
        data_path=args.data_path,
        output_metrics_path=args.output_metrics_path,
    )

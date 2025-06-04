import os
import time
from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import sklearn  # To get version

from src.utils.logger import get_logger
from src.utils.utils import save_json, load_json, get_timestamp_version

logger = get_logger(__name__)


class ColonyStrengthClassifier:
    """
    A classifier for determining colony strength, based on Scikit-learn's RandomForestClassifier.
    Handles model training, prediction, evaluation, and persistence with versioning.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the classifier with model and training configurations.
        Args:
            config (Dict[str, Any]): The main configuration dictionary.
        """
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})

        model_type = self.model_config.get("type", "RandomForestClassifier")
        if model_type != "RandomForestClassifier":
            # This is a simple implementation focusing on RandomForest.
            # For other model types, a factory or strategy pattern would be more suitable.
            logger.warning(
                f"Model type '{model_type}' specified, but this class primarily supports "
                f"RandomForestClassifier. Behavior might be unexpected for other types unless "
                f"they share a similar Scikit-learn interface and hyperparameters."
            )

        self.hyperparameters = self.model_config.get("hyperparameters", {})
        # Ensure 'random_state' is present for reproducibility if not in config
        if "random_state" not in self.hyperparameters:
            self.hyperparameters["random_state"] = 42  # Default random state
            logger.info(
                f"No random_state in config, defaulting to {self.hyperparameters['random_state']}"
            )

        self.model = RandomForestClassifier(**self.hyperparameters)

        self.model_save_path_base = self.model_config.get(
            "save_path", "artifacts/models"
        )
        self.version_strategy = self.model_config.get("version_strategy", "timestamp")
        self.model_version: Optional[str] = None
        self.trained_features: List[str] = []
        self.classes_: Optional[List[Any]] = None  # Store trained classes

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Trains the model on the provided feature set (X) and target (y).
        Performs a train-validation split for internal evaluation.

        Args:
            X (pd.DataFrame): DataFrame of features.
            y (pd.Series): Series of target labels.

        Returns:
            Dict[str, Any]: A dictionary containing training and validation metrics.
        """
        logger.info(
            f"Starting model training with hyperparameters: {self.hyperparameters}"
        )
        self.trained_features = X.columns.tolist()

        test_size = self.training_config.get("test_size", 0.2)
        stratify_target = self.training_config.get("stratify_by_target", True)

        # Use the model's random_state for the train_test_split for consistency
        split_random_state = self.hyperparameters.get("random_state", 42)

        should_stratify = stratify_target and y is not None and len(y.unique()) > 1
        if stratify_target and not should_stratify:
            logger.warning(
                "Stratification requested but cannot be performed (e.g. single class in y). Proceeding without stratification."
            )

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=split_random_state,
            stratify=y if should_stratify else None,
        )

        logger.info(
            f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}"
        )
        logger.info(
            f"Class distribution in training data (normalized):\n{y_train.value_counts(normalize=True).sort_index()}"
        )
        if y_val is not None and not y_val.empty:
            logger.info(
                f"Class distribution in validation data (normalized):\n{y_val.value_counts(normalize=True).sort_index()}"
            )

        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_.tolist()  # Store classes after fit
        logger.info("Model training completed.")

        # Evaluate on training and validation set
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)

        metrics = {}
        metrics_to_log = self.training_config.get(
            "metrics_to_log", ["accuracy", "f1_macro"]
        )

        if "accuracy" in metrics_to_log:
            metrics["train_accuracy"] = accuracy_score(y_train, train_preds)
            metrics["val_accuracy"] = accuracy_score(y_val, val_preds)
            logger.info(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
            logger.info(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")

        if "f1_macro" in metrics_to_log:
            metrics["train_f1_macro"] = f1_score(
                y_train, train_preds, average="macro", zero_division=0
            )
            metrics["val_f1_macro"] = f1_score(
                y_val, val_preds, average="macro", zero_division=0
            )
            logger.info(
                f"Train F1 Macro (zero_division=0): {metrics['train_f1_macro']:.4f}"
            )
            logger.info(
                f"Validation F1 Macro (zero_division=0): {metrics['val_f1_macro']:.4f}"
            )

        if "confusion_matrix" in metrics_to_log and self.classes_ is not None:
            cm = confusion_matrix(y_val, val_preds, labels=self.classes_)
            metrics["val_confusion_matrix"] = cm.tolist()
            logger.info(f"Validation Confusion Matrix (labels: {self.classes_}):\n{cm}")

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_importance_map = dict(zip(self.trained_features, importances))
            # Sort by importance, descending
            sorted_feature_importance = dict(
                sorted(
                    feature_importance_map.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            metrics["feature_importances"] = sorted_feature_importance
            logger.info("Feature Importances (top 5):")
            for i, (feat, imp) in enumerate(sorted_feature_importance.items()):
                if i < 5:
                    logger.info(f"  {feat}: {imp:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Makes predictions on new data."""
        if not self.trained_features:
            raise RuntimeError(
                "Model has not been trained or loaded properly (no trained_features). Call fit() or load_model() first."
            )
        if not hasattr(self.model, "classes_"):  # Check if model is fitted
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        try:
            # Ensure columns are in the same order and all trained features are present
            X_aligned = X[self.trained_features]
        except KeyError as e:
            missing_cols = set(self.trained_features) - set(X.columns)
            extra_cols = set(X.columns) - set(self.trained_features)
            error_msg = f"Feature mismatch during predict. Missing from input: {missing_cols}. Extra in input: {extra_cols}. Original error: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return self.model.predict(X_aligned)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Makes probability predictions on new data."""
        if not self.trained_features:
            raise RuntimeError(
                "Model has not been trained or loaded properly (no trained_features). Call fit() or load_model() first."
            )
        if not hasattr(self.model, "classes_"):
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        try:
            X_aligned = X[self.trained_features]
        except KeyError as e:
            missing_cols = set(self.trained_features) - set(X.columns)
            extra_cols = set(X.columns) - set(self.trained_features)
            error_msg = f"Feature mismatch during predict_proba. Missing: {missing_cols}. Extra: {extra_cols}. Original error: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return self.model.predict_proba(X_aligned)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluates the model on a given test set."""
        preds = self.predict(X)
        metrics = {}
        metrics_to_log = self.training_config.get(
            "metrics_to_log", ["accuracy", "f1_macro"]
        )

        if "accuracy" in metrics_to_log:
            metrics["test_accuracy"] = accuracy_score(y, preds)
        if "f1_macro" in metrics_to_log:
            metrics["test_f1_macro"] = f1_score(
                y, preds, average="macro", zero_division=0
            )
        if "confusion_matrix" in metrics_to_log and self.classes_ is not None:
            cm = confusion_matrix(y, preds, labels=self.classes_)
            metrics["test_confusion_matrix"] = cm.tolist()

        logger.info(f"Evaluation metrics on provided test set: {metrics}")
        return metrics

    def save_model(self, metrics: Optional[Dict[str, Any]] = None) -> str:
        """
        Saves the trained model and its metadata.
        Implements versioning strategy specified in config.

        Args:
            metrics (Optional[Dict[str, Any]]): Metrics dictionary from training/evaluation to save in metadata.

        Returns:
            str: The full path to the saved model file.
        """
        if self.version_strategy == "timestamp":
            self.model_version = get_timestamp_version()
        else:  # Default or other strategies
            self.model_version = get_timestamp_version()  # Fallback to timestamp
            logger.warning(
                f"Unknown version_strategy '{self.version_strategy}', defaulting to timestamp."
            )

        model_type_name = (
            self.model_config.get("type", "RandomForestClassifier")
            .lower()
            .replace("classifier", "")
        )
        model_filename = f"{model_type_name}_model_{self.model_version}.joblib"

        # Create full path for model saving
        os.makedirs(self.model_save_path_base, exist_ok=True)
        model_filepath = os.path.join(self.model_save_path_base, model_filename)

        joblib.dump(self.model, model_filepath)
        logger.info(f"Model saved to {model_filepath}")

        metadata = {
            "model_name": self.model_config.get("type", "RandomForestClassifier"),
            "model_version": self.model_version,
            "save_timestamp_utc": time.time(),  # UTC timestamp
            "hyperparameters": self.hyperparameters,
            "metrics": metrics or {},
            "trained_features": self.trained_features,
            "model_classes": self.classes_,
            "sklearn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        }

        metadata_filename = (
            f"{model_type_name}_model_{self.model_version}_metadata.json"
        )
        metadata_filepath = os.path.join(self.model_save_path_base, metadata_filename)
        save_json(metadata, metadata_filepath)
        logger.info(f"Model metadata saved to {metadata_filepath}")

        return model_filepath

    def load_model(self, model_filepath: str):
        """
        Loads a trained model and its metadata from specified paths.
        Args:
            model_filepath (str): Path to the .joblib model file.
        """
        if not os.path.exists(model_filepath):
            logger.error(f"Model file not found: {model_filepath}")
            raise FileNotFoundError(f"Model file not found: {model_filepath}")

        try:
            self.model = joblib.load(model_filepath)
            logger.info(f"Model loaded successfully from {model_filepath}")

            # Convention: metadata file is named similarly, replacing .joblib with _metadata.json
            metadata_filepath = model_filepath.replace(".joblib", "_metadata.json")
            if os.path.exists(metadata_filepath):
                metadata = load_json(metadata_filepath)
                self.model_version = metadata.get("model_version", "unknown_version")
                self.trained_features = metadata.get("trained_features", [])
                self.hyperparameters = metadata.get(
                    "hyperparameters", self.hyperparameters
                )  # Restore HPs
                self.classes_ = metadata.get("model_classes")
                logger.info(
                    f"Model metadata loaded from {metadata_filepath}. Model version: {self.model_version}, "
                    f"Trained features count: {len(self.trained_features)}, Classes: {self.classes_}"
                )
            else:
                logger.warning(
                    f"Metadata file not found at {metadata_filepath}. Some model attributes (like trained_features, version) might not be fully restored."
                )
                # Attempt to infer from loaded model if possible (less reliable)
                if hasattr(self.model, "feature_names_in_"):  # Scikit-learn 1.0+
                    self.trained_features = self.model.feature_names_in_.tolist()
                if hasattr(self.model, "classes_"):
                    self.classes_ = self.model.classes_.tolist()

        except Exception as e:
            logger.error(f"Error loading model from {model_filepath}: {e}")
            raise

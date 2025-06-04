import pytest
import pandas as pd
import os
from src.models.colony_classifier import ColonyStrengthClassifier
from src.data.preprocessor import Preprocessor  # To get sample X, y for model training
from src.utils.utils import load_json
import numpy as np
from pathlib import Path


# Helper to get sample X, y for model tests
@pytest.fixture
def sample_X_y(
    base_config: dict, sample_data_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, Preprocessor]:
    preprocessor = Preprocessor(config=base_config)
    # Use a small, consistent subset for model tests to ensure reproducibility and speed
    df_subset = sample_data_df.head(
        60
    ).copy()  # Ensure enough samples for train/val split

    # Ensure there are at least two classes in the target for stratification
    if df_subset[base_config["data"]["target_column"]].nunique() < 2:
        # If not, try a larger subset or skip if data is inherently single-class for this small slice
        df_subset = sample_data_df.head(200).copy()
        if df_subset[base_config["data"]["target_column"]].nunique() < 2:
            pytest.skip(
                "Sample data subset for model test has less than 2 classes for target."
            )

    X, y, _ = preprocessor.fit_transform(df_subset)

    if X.empty or y.empty or y.nunique() < 2:  # Final check after preprocessing
        pytest.skip(
            "Preprocessing resulted in empty/unsuitable data for model testing (e.g. <2 classes)."
        )
    return X, y, preprocessor  # Return preprocessor for its state if needed


def test_model_init(base_config: dict):
    """Test ColonyStrengthClassifier initialization."""
    model_trainer = ColonyStrengthClassifier(config=base_config)
    assert model_trainer.model is not None
    assert model_trainer.hyperparameters == base_config["model"]["hyperparameters"]
    assert (
        model_trainer.hyperparameters["random_state"] is not None
    )  # Ensure random_state is set


def test_model_fit(
    base_config: dict, sample_X_y: tuple[pd.DataFrame, pd.Series, Preprocessor]
):
    """Test model fitting process and metrics generation."""
    X, y, _ = sample_X_y
    model_trainer = ColonyStrengthClassifier(config=base_config)
    metrics = model_trainer.fit(X, y)

    assert "train_accuracy" in metrics
    assert "val_accuracy" in metrics
    assert metrics["train_accuracy"] >= 0 and metrics["train_accuracy"] <= 1
    assert metrics["val_accuracy"] >= 0 and metrics["val_accuracy"] <= 1
    assert model_trainer.trained_features == list(X.columns)
    assert model_trainer.classes_ is not None
    assert len(model_trainer.classes_) == y.nunique()


def test_model_predict_and_predict_proba(
    base_config: dict, sample_X_y: tuple[pd.DataFrame, pd.Series, Preprocessor]
):
    """Test model prediction and probability prediction."""
    X, y, _ = sample_X_y
    model_trainer = ColonyStrengthClassifier(config=base_config)
    model_trainer.fit(X, y)  # Fit the model first

    # Test predict
    predictions = model_trainer.predict(X.head(5))
    assert len(predictions) == 5
    assert all(
        item in model_trainer.classes_ for item in predictions
    )  # Predictions should be valid classes

    # Test predict_proba
    probabilities = model_trainer.predict_proba(X.head(5))
    assert probabilities.shape == (5, len(model_trainer.classes_))
    assert np.allclose(
        probabilities.sum(axis=1), 1.0
    )  # Probabilities for each sample should sum to 1


def test_model_evaluate(
    base_config: dict, sample_X_y: tuple[pd.DataFrame, pd.Series, Preprocessor]
):
    """Test model evaluation on a test set."""
    X, y, _ = sample_X_y
    model_trainer = ColonyStrengthClassifier(config=base_config)
    model_trainer.fit(X, y)  # Fit the model

    # Use a portion of X, y as a pseudo "test set" for evaluation method
    X_test_sample, y_test_sample = X.tail(10), y.tail(10)
    if X_test_sample.empty:
        pytest.skip("Not enough data for test sample in evaluate.")

    eval_metrics = model_trainer.evaluate(X_test_sample, y_test_sample)
    assert "test_accuracy" in eval_metrics
    assert eval_metrics["test_accuracy"] >= 0 and eval_metrics["test_accuracy"] <= 1
    if "f1_macro" in base_config["training"]["metrics_to_log"]:
        assert "test_f1_macro" in eval_metrics


def test_model_save_load(
    base_config: dict,
    sample_X_y: tuple[pd.DataFrame, pd.Series, Preprocessor],
    test_artifacts_dir: Path,
):
    """Test saving and loading a trained model and its metadata."""
    X, y, _ = sample_X_y

    # Ensure model save path in config points to the test artifacts directory
    config_for_save_load = base_config.copy()
    model_save_dir = test_artifacts_dir / "models_save_load_test"  # Unique sub-dir
    config_for_save_load["model"]["save_path"] = str(model_save_dir)

    original_model = ColonyStrengthClassifier(config=config_for_save_load)
    fit_metrics = original_model.fit(X, y)

    model_filepath = original_model.save_model(metrics=fit_metrics)
    assert os.path.exists(model_filepath)

    metadata_filepath = model_filepath.replace(".joblib", "_metadata.json")
    assert os.path.exists(metadata_filepath)
    loaded_meta = load_json(metadata_filepath)
    assert loaded_meta["model_version"] == original_model.model_version
    assert loaded_meta["trained_features"] == original_model.trained_features

    # Load the model into a new instance
    loaded_model = ColonyStrengthClassifier(config=config_for_save_load)
    loaded_model.load_model(model_filepath)

    assert loaded_model.model is not None
    assert loaded_model.model_version == original_model.model_version
    assert loaded_model.trained_features == original_model.trained_features
    assert loaded_model.classes_ == original_model.classes_
    assert loaded_model.hyperparameters == original_model.hyperparameters

    # Check if the loaded model can make predictions
    predictions = loaded_model.predict(X.head(5))
    assert len(predictions) == 5
    # Compare with original model's predictions if model is deterministic
    original_predictions = original_model.predict(X.head(5))
    assert np.array_equal(predictions, original_predictions)


def test_model_unfitted_predict_error(
    base_config: dict, sample_X_y: tuple[pd.DataFrame, pd.Series, Preprocessor]
):
    """Test that predict raises RuntimeError if model is not fitted."""
    X, _, _ = sample_X_y
    unfitted_model = ColonyStrengthClassifier(config=base_config)  # Not fitted

    # Need to set trained_features manually to bypass the first check for this specific test
    unfitted_model.trained_features = X.columns.tolist()

    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        unfitted_model.predict(X.head(1))

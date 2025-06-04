from pathlib import Path

# Assuming src.train.run_training is the main entry point for a single training run
from src.train import run_training
from src.utils.utils import load_json


def test_full_training_pipeline_run_successful(
    temp_config_file: str, base_config: dict, test_artifacts_dir: Path
):
    """
    Integration test for the main training pipeline (run_training function).
    Uses a temporary config file that points artifacts to a test-specific directory.
    """
    # temp_config_file fixture provides a path to a config.yaml populated with base_config,
    # where base_config itself uses test_artifacts_dir for output paths.

    # Ensure the directories for artifacts (as specified in base_config) are clear or exist
    # (pytest tmp_path fixtures handle creation and cleanup of test_artifacts_dir)
    model_save_path = Path(base_config["model"]["save_path"])
    metrics_base_path = Path(
        test_artifacts_dir / "metrics"
    )  # From base_config structure
    metrics_base_path.mkdir(parents=True, exist_ok=True)

    # Execute the training pipeline
    run_training(config_path=temp_config_file)

    # --- Assertions for Artifacts ---

    # 1. Check for Model File and Metadata
    assert model_save_path.exists()
    model_files = list(model_save_path.glob("*.joblib"))
    assert len(model_files) >= 1, "No .joblib model file was created."

    # Assuming one model is created per run for this test config
    model_file = model_files[0]
    metadata_file = Path(str(model_file).replace(".joblib", "_metadata.json"))
    assert (
        metadata_file.exists()
    ), f"Metadata file {metadata_file.name} not found for model {model_file.name}."

    # Optionally, load and check some metadata content
    metadata_content = load_json(metadata_file)
    assert "model_version" in metadata_content
    assert "hyperparameters" in metadata_content
    assert "metrics" in metadata_content
    assert "trained_features" in metadata_content
    assert len(metadata_content["trained_features"]) > 0

    # 2. Check for Preprocessor State File
    # `run_training` saves preprocessor_state.json in the model_save_path directory.
    preprocessor_state_file = model_save_path / "preprocessor_state.json"
    assert (
        preprocessor_state_file.exists()
    ), "Preprocessor state file (preprocessor_state.json) not found."

    preprocessor_state_content = load_json(preprocessor_state_file)
    assert "fit_params" in preprocessor_state_content
    assert "base_features" in preprocessor_state_content
    assert len(preprocessor_state_content["base_features"]) > 0

    # 3. Check for Metrics Files
    # Latest metrics file
    latest_metrics_file = Path(base_config["training"]["metrics_file_path"])
    assert (
        latest_metrics_file.exists()
    ), "Latest metrics file (e.g., latest_metrics.json) not found."
    latest_metrics_content = load_json(latest_metrics_file)
    assert "val_accuracy" in latest_metrics_content  # A key metric
    assert (
        latest_metrics_content["val_accuracy"] >= 0
        and latest_metrics_content["val_accuracy"] <= 1
    )

    # Experiment log file
    experiment_log_file = Path(base_config["training"]["experiment_tracking_file"])
    assert (
        experiment_log_file.exists()
    ), "Experiment log file (e.g., experiment_log.json) not found."
    experiment_log_content = load_json(experiment_log_file)
    assert isinstance(experiment_log_content, list), "Experiment log should be a list."
    assert len(experiment_log_content) >= 1, "Experiment log is empty."

    # Check content of the last (current) experiment entry
    last_experiment = experiment_log_content[-1]
    assert "run_timestamp" in last_experiment
    assert "model_path" in last_experiment
    assert last_experiment["model_path"].endswith(model_file.name)
    assert "metrics" in last_experiment
    assert "val_accuracy" in last_experiment["metrics"]
    assert "features_used" in last_experiment
    assert len(last_experiment["features_used"]) > 0
    assert "preprocessor_state_path" in last_experiment
    assert last_experiment["preprocessor_state_path"].endswith(
        "preprocessor_state.json"
    )


# Add more integration tests if needed, e.g., one that simulates a failure condition.
# For instance, providing a config that points to non-existent data for the DataLoader
# and asserting that run_training handles this gracefully (e.g. raises specific error or logs and exits).

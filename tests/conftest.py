import pytest
import pandas as pd
import yaml
from pathlib import Path


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_data_path(project_root: Path) -> Path:
    """Path to the sample CSV data file."""
    return project_root / "data" / "colony_size.csv"


@pytest.fixture(scope="session")
def sample_data_df(sample_data_path: Path) -> pd.DataFrame:
    """Loads the sample data CSV into a DataFrame."""
    if not sample_data_path.exists():
        pytest.fail(f"Sample data file not found at {sample_data_path}")
    return pd.read_csv(sample_data_path)


@pytest.fixture(scope="module")
def test_artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Creates a temporary directory for test artifacts for the module."""
    # tmp_path_factory creates a unique temp dir per test invocation of the fixture (if session scoped)
    # or per module if module scoped.
    # This path is automatically cleaned up by pytest.
    return tmp_path_factory.mktemp("test_artifacts_module")


@pytest.fixture
def base_config(sample_data_path: Path, test_artifacts_dir: Path) -> dict:
    """Provides a base configuration dictionary for tests, using temp paths for artifacts."""
    # Using .resolve().as_posix() for cross-platform compatibility in paths within YAML string
    config_content = f"""
data:
  sources:
    - type: csv
      path: {sample_data_path.resolve().as_posix()}
  target_column: "size"
  validation:
    check_nulls: true
    required_columns: ["sensor_id", "temperature_sensor", "size"] # Minimal set for most tests
    sensor_ranges:
      temperature_sensor: [0, 60]

preprocessing:
  fillna_strategy: "mean"
  temp_sensors_cols: ["temperature_sensor", "temperature_gateway"]
  feature_engineering:
    base_temp_cols: ["temperature_sensor", "temperature_gateway"]
    transmission_col: "ihs_to_gw_transmission_strength"
  outlier_quantiles:
    temperature_sensor_mean: [0.01, 0.99]

model:
  type: RandomForestClassifier
  hyperparameters:
    n_estimators: 5 # Very small for fast tests
    max_depth: 2    # Very shallow
    random_state: 42
  save_path: {test_artifacts_dir.joinpath("models").resolve().as_posix()}
  version_strategy: "timestamp"

training:
  test_size: 0.3 # Slightly larger test size for robust metric calculation in small test data
  stratify_by_target: true
  metrics_to_log: ['accuracy', 'f1_macro'] # 'confusion_matrix' can be large for logs
  experiment_tracking_file: {test_artifacts_dir.joinpath("metrics/experiment_log.json").resolve().as_posix()}
  metrics_file_path: {test_artifacts_dir.joinpath("metrics/latest_metrics.json").resolve().as_posix()}

evaluation: # For evaluate.py tests if any
  metrics_path: {test_artifacts_dir.joinpath("metrics_eval").resolve().as_posix()}
"""
    return yaml.safe_load(config_content)


@pytest.fixture
def temp_config_file(tmp_path: Path, base_config: dict) -> str:
    """Creates a temporary YAML config file from the base_config and returns its path string."""
    # tmp_path is a fixture providing a temporary directory unique to each test function.
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(base_config, f)
    return str(config_file)

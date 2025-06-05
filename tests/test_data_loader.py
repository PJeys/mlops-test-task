import pytest
import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader


def test_data_loader_init(base_config: dict):
    """Test DataLoader initialization."""
    loader = DataLoader(config=base_config)
    assert loader.config == base_config["data"]
    assert len(loader.sources) > 0
    assert loader.sources[0]["type"] == "csv"


def test_data_loader_load_csv_success(base_config: dict, sample_data_df: pd.DataFrame):
    """Test successful loading of CSV data."""
    loader = DataLoader(config=base_config)
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # Compare shapes; sample_data_df is fixture loaded from same path as in base_config
    pd.testing.assert_frame_equal(
        df, sample_data_df, check_dtype=False
    )  # Dtype can vary slightly


def test_data_loader_validation_required_cols_pass(base_config: dict):
    """Test validation passes when all required columns are present."""
    loader = DataLoader(config=base_config)
    loader.load_data()  # This internally calls _validate_data
    # No error means pass


def test_data_loader_validation_required_cols_fail(
    base_config: dict, sample_data_df: pd.DataFrame
):
    """Test validation fails if a required column is missing."""
    faulty_df = sample_data_df.drop(
        columns=["sensor_id"]
    )  # sensor_id is required in base_config
    loader = DataLoader(config=base_config)
    with pytest.raises(
        ValueError, match=r"DataFrame is missing required columns: \['sensor_id'\]"
    ):
        loader._validate_data(faulty_df)


def test_data_loader_validation_null_check_warning(base_config: dict, caplog):
    """Test null check warning when check_nulls is true and nulls exist."""
    config_with_null_check = base_config.copy()
    config_with_null_check["data"]["validation"]["check_nulls"] = True
    loader = DataLoader(config=config_with_null_check)

    # Create a DataFrame with a null value in a required column
    df_with_nulls = pd.DataFrame(
        {
            "sensor_id": [1, 2, None, 4],  # Null in a required column
            "temperature_sensor": [25, 26, 27, 28],
            "size": ["S", "M", "L", "S"],
        }
    )
    # Add other required columns from base_config to pass initial checks before null check
    for req_col in config_with_null_check["data"]["validation"].get(
        "required_columns", []
    ):
        if req_col not in df_with_nulls.columns:
            df_with_nulls[req_col] = [1, 2, 3, 4]  # Dummy data

    loader._validate_data(df_with_nulls)
    assert "Null values found" in caplog.text
    assert "sensor_id" in caplog.text  # Column with null


def test_data_loader_sensor_range_check_warning(base_config: dict, caplog):
    """Test sensor range check warning for values outside specified range."""
    config_with_range_check = base_config.copy()
    # Tighten range for temperature_sensor for testing
    config_with_range_check["data"]["validation"]["sensor_ranges"] = {
        "temperature_sensor": [10, 30]
    }
    loader = DataLoader(config=config_with_range_check)

    df_out_of_range = pd.DataFrame(
        {
            "sensor_id": [1, 2, 3],
            "temperature_sensor": [5, 25, 35],  # 5 (below min) and 35 (above max)
            "size": ["S", "M", "L"],
        }
    )
    # Add other required columns
    for req_col in config_with_range_check["data"]["validation"].get(
        "required_columns", []
    ):
        if req_col not in df_out_of_range.columns:
            df_out_of_range[req_col] = [1, 2, 3]

    loader._validate_data(df_out_of_range)
    assert "values outside the expected range" in caplog.text
    assert "temperature_sensor" in caplog.text
    assert (
        "5" in caplog.text or "35" in caplog.text
    )  # Check if an out-of-range value is mentioned


def test_data_loader_file_not_found_error(base_config: dict, tmp_path: Path):
    """Test FileNotFoundError when primary data file and backup are not found."""
    config_bad_path = base_config.copy()
    non_existent_file = tmp_path / "this_file_does_not_exist.csv"
    config_bad_path["data"]["sources"][0]["path"] = str(non_existent_file)

    loader = DataLoader(config=config_bad_path)
    with pytest.raises(FileNotFoundError):
        loader.load_data()


def test_data_loader_backup_file_logic(
    base_config: dict, tmp_path: Path, sample_data_df: pd.DataFrame
):
    """Test that backup file is used if primary is not found."""
    config_backup_test = base_config.copy()

    primary_path = tmp_path / "primary_data.csv"  # This will not exist
    backup_file_name = (
        "backup_primary_data.csv"  # Matches DataLoader's backup convention
    )
    backup_path = tmp_path / backup_file_name

    # Save sample data to the backup path
    sample_data_df.to_csv(backup_path, index=False)

    config_backup_test["data"]["sources"][0]["path"] = str(primary_path)

    loader = DataLoader(config=config_backup_test)
    df_loaded = loader.load_data()  # Should load from backup_path

    assert not df_loaded.empty
    pd.testing.assert_frame_equal(df_loaded, sample_data_df, check_dtype=False)


def test_data_loader_data_hash(base_config: dict):
    """Test that a data hash is generated."""
    loader = DataLoader(config=base_config)
    df = loader.load_data()  # This calls _get_data_hash
    # The hash itself is tricky to assert directly without re-implementing.
    # For now, check that it's a non-empty string (assuming it's logged).
    # Better: make _get_data_hash public for direct testing or check log output.

    # For direct test:
    hasher = loader._get_data_hash(df)
    assert isinstance(hasher, str)
    assert len(hasher) == 64  # SHA256 hex digest length

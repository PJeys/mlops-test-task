import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import Preprocessor


# Minimal DataFrame for testing specific preprocessor functions
@pytest.fixture
def small_test_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor_id": [1, 1, 2, 2, 3, 3],
            "temperature_sensor": [20, 22, 30, np.nan, 25, 26],
            "temperature_gateway": [18, np.nan, 28, 30, 23, 24],
            "ihs_to_gw_transmission_strength": [5, 7, 12, 15, np.nan, 10],
            "size": ["S", "S", "M", "M", "L", "L"],  # Target column
        }
    )


def test_preprocessor_init(base_config: dict):
    """Test Preprocessor initialization."""
    preprocessor = Preprocessor(config=base_config)
    assert preprocessor.config == base_config["preprocessing"]
    assert preprocessor.target_column == base_config["data"]["target_column"]


def test_fillna_numeric(base_config: dict, small_test_df: pd.DataFrame):
    """Test _fillna method for numeric columns with mean strategy."""
    preprocessor = Preprocessor(config=base_config)
    preprocessor.config["fillna_strategy"] = "mean"

    cols_to_fill = ["temperature_sensor", "temperature_gateway"]
    df_filled = preprocessor._fillna(
        small_test_df.copy(), cols_to_fill, is_fitting=True
    )

    assert not df_filled["temperature_sensor"].isnull().any()
    assert not df_filled["temperature_gateway"].isnull().any()
    expected_mean_temp_sensor = small_test_df['temperature_sensor'].mean()
    assert df_filled.loc[3, "temperature_sensor"] == pytest.approx(
        expected_mean_temp_sensor
    )
    assert "temperature_sensor_fill_value" in preprocessor._fit_params


def test_engineer_features_temperature(base_config: dict, small_test_df: pd.DataFrame):
    """Test temperature feature engineering aspects."""
    preprocessor = Preprocessor(config=base_config)
    df_engineered = preprocessor._engineer_features(
        small_test_df.copy(), is_fitting=True
    )

    assert "temperature_sensor_mean" in df_engineered.columns
    assert "temperature_sensor_std" in df_engineered.columns
    assert "temperature_sensor_min" in df_engineered.columns
    assert "temperature_sensor_max" in df_engineered.columns
    assert "temperature_sensor_range" in df_engineered.columns

    # Check values for sensor_id = 1 (temp_sensor values: 20, 22)
    sensor1_data = df_engineered[df_engineered["sensor_id"] == 1]
    assert sensor1_data["temperature_sensor_mean"].iloc[0] == pytest.approx(21)
    assert sensor1_data["temperature_sensor_range"].iloc[0] == pytest.approx(
        2
    )  # 22 - 20

    assert "temperature_sensor_grouped_stats" in preprocessor._fit_params


def test_engineer_features_transmission(base_config: dict, small_test_df: pd.DataFrame):
    """Test transmission feature engineering aspects."""
    preprocessor = Preprocessor(config=base_config)
    # First, fill NaNs in transmission col as it's a base for engineering
    df_temp_filled = preprocessor._fillna(
        small_test_df.copy(), ["ihs_to_gw_transmission_strength"], is_fitting=True
    )
    df_engineered = preprocessor._engineer_features(df_temp_filled, is_fitting=True)

    transmission_col = base_config["preprocessing"]["feature_engineering"][
        "transmission_col"
    ]
    assert f"{transmission_col}_high_signal" in df_engineered.columns
    assert f"{transmission_col}_mean_by_sensor" in df_engineered.columns

    # Sensor 2: ihs_to_gw_transmission_strength values are 12, 15. Mean by sensor should be 13.5
    sensor2_data = df_engineered[df_engineered["sensor_id"] == 2]
    assert sensor2_data[f"{transmission_col}_mean_by_sensor"].iloc[0] == pytest.approx(
        13.5
    )
    assert f"{transmission_col}_grouped_stats" in preprocessor._fit_params


def test_remove_outliers(base_config: dict):
    """Test outlier removal logic."""
    preprocessor = Preprocessor(config=base_config)
    # Configure outlier removal for a hypothetical 'feature_x'
    preprocessor.config["outlier_quantiles"] = {"feature_x": [0.1, 0.9]}

    df_with_outliers = pd.DataFrame(
        {"feature_x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]}
    )  # 100 is outlier
    df_filtered = preprocessor._remove_outliers(
        df_with_outliers.copy(), is_fitting=True
    )

    assert 100 not in df_filtered["feature_x"].values
    assert len(df_filtered) < len(df_with_outliers)
    assert "feature_x_lower_bound" in preprocessor._fit_params
    assert "feature_x_upper_bound" in preprocessor._fit_params


def test_fit_transform_produces_output(base_config: dict, sample_data_df: pd.DataFrame):
    """Test the main fit_transform method for basic execution and output types."""
    preprocessor = Preprocessor(config=base_config)
    # Use a subset of sample data for speed
    df_subset = sample_data_df.head(100).copy()

    X, y, features = preprocessor.fit_transform(df_subset)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(features, list)
    assert not X.empty
    assert not y.empty
    assert len(features) > 0
    assert X.shape[0] == y.shape[0]
    assert (
        X.isnull().sum().sum() == 0
    ), "Features should not have NaNs after fit_transform"
    assert (
        y.isnull().sum().sum() == 0
    ), "Target should not have NaNs after fit_transform"
    assert all(feat_col in X.columns for feat_col in features)
    assert (
        base_config["data"]["target_column"] not in X.columns
    )  # Target should be separated
    assert len(preprocessor._fit_params) > 0, "Fit parameters should be populated"
    assert len(preprocessor.base_features) > 0, "Base features list should be populated"

def test_preprocessor_handles_missing_cols_in_config(
    base_config: dict, sample_data_df: pd.DataFrame
):
    """Test preprocessor gracefully handles if a configured column (e.g. for FE) is missing from data."""
    preprocessor = Preprocessor(config=base_config)

    # Data missing 'ihs_to_gw_transmission_strength' which is in config for feature_engineering
    df_missing_transmission = sample_data_df.drop(
        columns=["ihs_to_gw_transmission_strength"], errors="ignore"
    )

    # Should run without error, and transmission-related features won't be created.
    X, y, features = preprocessor.fit_transform(df_missing_transmission)

    transmission_col_config = base_config["preprocessing"]["feature_engineering"][
        "transmission_col"
    ]
    assert f"{transmission_col_config}_high_signal" not in X.columns
    assert f"{transmission_col_config}_mean_by_sensor" not in X.columns
    assert not X.empty

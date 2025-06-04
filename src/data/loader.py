import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import hashlib

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Handles loading and initial validation of data from various sources.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataLoader with data configuration.
        Args:
            config (Dict[str, Any]): The main configuration dictionary, expected
                                     to contain a 'data' key.
        """
        self.config = config.get("data", {})
        if not self.config:
            logger.warning(
                "Data configuration not found. DataLoader may not function correctly."
            )
        self.sources: List[Dict[str, Any]] = self.config.get("sources", [])
        self.validation_rules: Dict[str, Any] = self.config.get("validation", {})

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the primary configured source.
        Supports only one CSV source as per initial requirements.
        Performs data validation if rules are specified.

        Returns:
            pd.DataFrame: The loaded and validated DataFrame.

        Raises:
            ValueError: If no data sources are configured or if critical errors occur during loading.
            FileNotFoundError: If the data file cannot be found.
            NotImplementedError: If an unsupported data source type is specified.
        """
        if not self.sources:
            logger.error("No data sources configured.")
            raise ValueError("No data sources configured.")

        # Assuming a single primary source for now
        source_config = self.sources[0]
        source_type = source_config.get("type")

        if source_type != "csv":
            logger.error(f"Unsupported data source type: {source_type}")
            raise NotImplementedError(
                f"Data source type {source_type} not implemented."
            )

        file_path_str = source_config.get("path")
        if not file_path_str:
            logger.error("CSV file path not specified in config.")
            raise ValueError("CSV file path not specified.")

        file_path = Path(file_path_str)

        # Handle legacy backup path logic (simplified from original script)
        if not file_path.exists():
            logger.warning(
                f"Data file not found at {file_path}. Checking for legacy backup."
            )
            # Construct a potential backup path based on a simple convention
            # Example: 'data/colony_size.csv' -> 'data/backup_colony_size.csv'
            backup_path = file_path.parent / f"backup_{file_path.name}"
            if backup_path.exists():
                logger.info(f"Using backup data file: {backup_path}")
                file_path = backup_path
            else:
                logger.error(
                    f"Data file not found at {file_path_str} and no backup found at {backup_path}."
                )
                raise FileNotFoundError(
                    f"Data file {file_path_str} not found and no backup available."
                )

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")

            data_hash = self._get_data_hash(df)
            logger.info(
                f"Data version (content hash): {data_hash} for file: {file_path}"
            )

            self._validate_data(df)
            return df
        except Exception as e:
            logger.error(f"Error loading or validating data from {file_path}: {e}")
            raise

    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """
        Generates a SHA256 hash of the DataFrame's content for simple version tracking.
        Args:
            df (pd.DataFrame): The DataFrame to hash.
        Returns:
            str: A SHA256 hash string, or "unknown" if hashing fails.
        """
        try:
            # pd.util.hash_pandas_object can be slow for large DFs.
            # Hashing the raw file bytes before loading might be more performant for very large files.
            return hashlib.sha256(
                pd.util.hash_pandas_object(df, index=True).values
            ).hexdigest()
        except Exception as e:
            logger.warning(f"Could not generate data hash: {e}")
            return "unknown"

    def _validate_data(self, df: pd.DataFrame):
        """
        Validates the loaded DataFrame based on rules in the configuration.
        Args:
            df (pd.DataFrame): The DataFrame to validate.
        Raises:
            ValueError: If critical validation checks (like missing required columns) fail.
        """
        if not self.validation_rules:
            logger.info("No validation rules defined. Skipping data validation.")
            return

        logger.info("Starting data validation...")

        required_cols = self.validation_rules.get("required_columns", [])
        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                raise ValueError(
                    f"DataFrame is missing required columns: {missing_cols}"
                )

        if self.validation_rules.get("check_nulls", False):
            null_counts = df.isnull().sum()
            null_cols_summary = null_counts[null_counts > 0]
            if not null_cols_summary.empty:
                logger.warning(f"Null values found in columns:\n{null_cols_summary}")
            else:
                logger.info(
                    "No null values found in any columns (as per check_nulls config)."
                )

        sensor_ranges = self.validation_rules.get("sensor_ranges", {})
        for col, limits in sensor_ranges.items():
            if col in df.columns:
                min_val, max_val = limits
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Filter out NaNs before range check to avoid warnings/errors with comparison operators
                    valid_data = df[col].dropna()
                    outside_range = valid_data[
                        (valid_data < min_val) | (valid_data > max_val)
                    ]
                    if not outside_range.empty:
                        logger.warning(
                            f"Column '{col}' has {len(outside_range)} values outside the expected range "
                            f"[{min_val}, {max_val}]. Example out-of-range values: {outside_range.head(3).tolist()}"
                        )
                else:
                    logger.warning(
                        f"Column '{col}' for range check is not numeric. Skipping range check."
                    )
            else:
                logger.debug(
                    f"Column '{col}' specified for range check not found in DataFrame. Skipping."
                )

        logger.info("Data validation completed.")

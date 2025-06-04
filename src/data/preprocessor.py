import pandas as pd
from typing import List, Dict, Any, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Handles data preprocessing, including cleaning, feature engineering, and outlier removal.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Preprocessor with preprocessing and data configurations.
        Args:
            config (Dict[str, Any]): The main configuration dictionary.
        """
        self.config = config.get("preprocessing", {})
        self.feature_eng_config = self.config.get("feature_engineering", {})
        self.data_config = config.get("data", {})
        self.target_column = self.data_config.get("target_column", "size")

        self.base_features: List[str] = []
        self._fit_params: Dict[str, Any] = (
            {}
        )  # Stores learned parameters (e.g., means, quantiles)

    def _fillna(
        self, df: pd.DataFrame, columns: List[str], is_fitting: bool = True
    ) -> pd.DataFrame:
        """Fills missing values in specified columns."""
        strategy = self.config.get("fillna_strategy", "mean")
        df_processed = df.copy()

        for col in columns:
            if col in df_processed.columns:
                if df_processed[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        if is_fitting:  # Learn and store parameters
                            if strategy == "mean":
                                fill_value = df_processed[col].mean()
                                self._fit_params[f"{col}_fill_value"] = fill_value
                            elif strategy == "median":
                                fill_value = df_processed[col].median()
                                self._fit_params[f"{col}_fill_value"] = fill_value
                            # Add other strategies (mode, constant) if needed
                            else:
                                logger.warning(
                                    f"Unsupported numeric fillna strategy: {strategy} for column {col}. Using mean."
                                )
                                fill_value = df_processed[col].mean()
                                self._fit_params[f"{col}_fill_value"] = fill_value
                        else:  # Apply stored parameters
                            fill_value = self._fit_params.get(f"{col}_fill_value")
                            if (
                                fill_value is None
                            ):  # Fallback if param not found (e.g. new col during transform)
                                logger.warning(
                                    f"Fit parameter for '{col}_fill_value' not found during transform. Using current data's mean."
                                )
                                fill_value = df_processed[
                                    col
                                ].mean()  # Risky, ideally should error or have robust fallback

                        df_processed[col] = df_processed[col].fillna(fill_value)
                        logger.debug(
                            f"Filled NaNs in '{col}' with {strategy}: {fill_value:.4f}"
                        )
                    else:  # For non-numeric, mode is a common strategy
                        if is_fitting:
                            fill_value = (
                                df_processed[col].mode()[0]
                                if not df_processed[col].mode().empty
                                else "Unknown"
                            )
                            self._fit_params[f"{col}_fill_value"] = fill_value
                        else:
                            fill_value = self._fit_params.get(
                                f"{col}_fill_value",
                                (
                                    df_processed[col].mode()[0]
                                    if not df_processed[col].mode().empty
                                    else "Unknown"
                                ),
                            )
                        df_processed[col] = df_processed[col].fillna(fill_value)
                        logger.debug(
                            f"Filled NaNs in non-numeric '{col}' with mode: {fill_value}"
                        )
        return df_processed

    def _engineer_features(
        self, df: pd.DataFrame, is_fitting: bool = True
    ) -> pd.DataFrame:
        """Performs feature engineering based on configuration."""
        logger.info("Starting feature engineering...")
        df_processed = df.copy()
        group_key = (
            "sensor_id"  # Assuming sensor_id is always the key for these aggregations
        )

        # Temperature features
        temp_cols_to_process = [
            col
            for col in self.feature_eng_config.get("base_temp_cols", [])
            if col in df_processed.columns
        ]

        if group_key not in df_processed.columns:
            logger.warning(
                f"`{group_key}` column not found. Skipping sensor_id based aggregations."
            )
        elif not temp_cols_to_process:
            logger.warning("No base temperature columns found for feature engineering.")
        else:
            agg_ops = ["mean", "std", "min", "max"]
            for base_col_prefix in temp_cols_to_process:
                if base_col_prefix not in df_processed.columns:
                    logger.warning(
                        f"Base column {base_col_prefix} for temp aggregation not found. Skipping."
                    )
                    continue

                # Grouped aggregations
                if is_fitting:
                    grouped_stats = (
                        df_processed.groupby(group_key)[base_col_prefix]
                        .agg(agg_ops)
                        .add_prefix(f"{base_col_prefix}_")
                    )
                    self._fit_params[f"{base_col_prefix}_grouped_stats"] = (
                        grouped_stats  # Store entire aggregated df
                    )
                else:
                    grouped_stats = self._fit_params.get(
                        f"{base_col_prefix}_grouped_stats"
                    )
                    if grouped_stats is None:
                        logger.warning(
                            f"Fit parameter for '{base_col_prefix}_grouped_stats'"
                            f" not found during transform. Recomputing on current data"
                            f" (may lead to data leakage or errors)."
                        )
                        # Fallback: recompute on current data (not ideal for transform)
                        # This path might be problematic if new sensor_ids appear in transform data.
                        # A robust solution would handle new sensor_ids (e.g., fill with global mean/median from training).
                        grouped_stats = (
                            df_processed.groupby(group_key)[base_col_prefix]
                            .agg(agg_ops)
                            .add_prefix(f"{base_col_prefix}_")
                        )

                df_processed = df_processed.merge(
                    grouped_stats, on=group_key, how="left", suffixes=("", "_drop")
                )
                # Clean up potential duplicate columns from merge if original column had same name as an aggregated one (unlikely with prefix)
                df_processed = df_processed[
                    [c for c in df_processed.columns if not c.endswith("_drop")]
                ]

                # Range feature
                min_col = f"{base_col_prefix}_min"
                max_col = f"{base_col_prefix}_max"
                range_col = f"{base_col_prefix}_range"
                if min_col in df_processed.columns and max_col in df_processed.columns:
                    df_processed[range_col] = (
                        df_processed[max_col] - df_processed[min_col]
                    )
                else:
                    logger.warning(
                        f"Min/max columns for {base_col_prefix} not found after aggregation. Cannot create range feature."
                    )

        # Transmission strength features
        transmission_col = self.feature_eng_config.get("transmission_col")
        if transmission_col and transmission_col in df_processed.columns:
            df_processed[f"{transmission_col}_high_signal"] = (
                df_processed[transmission_col] > 10
            ).astype(
                int
            )  # Example threshold from legacy

            if group_key in df_processed.columns:
                if is_fitting:
                    transmission_grouped_stats = df_processed.groupby(group_key)[
                        transmission_col
                    ].agg(["mean", "std"])
                    transmission_grouped_stats.columns = [
                        f"{transmission_col}_mean_by_sensor",
                        f"{transmission_col}_std_by_sensor",
                    ]
                    self._fit_params[f"{transmission_col}_grouped_stats"] = (
                        transmission_grouped_stats
                    )
                else:
                    transmission_grouped_stats = self._fit_params.get(
                        f"{transmission_col}_grouped_stats"
                    )
                    if transmission_grouped_stats is None:
                        logger.warning(
                            f"Fit parameter for '{transmission_col}_grouped_stats' not found. Recomputing on current data."
                        )
                        transmission_grouped_stats = df_processed.groupby(group_key)[
                            transmission_col
                        ].agg(["mean", "std"])
                        transmission_grouped_stats.columns = [
                            f"{transmission_col}_mean_by_sensor",
                            f"{transmission_col}_std_by_sensor",
                        ]

                df_processed = df_processed.merge(
                    transmission_grouped_stats,
                    on=group_key,
                    how="left",
                    suffixes=("", "_drop"),
                )
                df_processed = df_processed[
                    [c for c in df_processed.columns if not c.endswith("_drop")]
                ]

                mean_col_by_sensor = f"{transmission_col}_mean_by_sensor"
                if mean_col_by_sensor in df_processed.columns:
                    df_processed[f"{transmission_col}_deviation_from_sensor_mean"] = (
                        abs(
                            df_processed[transmission_col]
                            - df_processed[mean_col_by_sensor]
                        )
                    )
        else:
            logger.debug(
                f"Transmission column '{transmission_col}' not found or not configured. Skipping transmission features."
            )

        logger.info("Feature engineering completed.")
        return df_processed

    def _remove_outliers(
        self, df: pd.DataFrame, is_fitting: bool = True
    ) -> pd.DataFrame:
        """Removes outliers based on configured quantiles for specified columns."""
        df_processed = df.copy()
        outlier_rules = self.config.get("outlier_quantiles", {})

        for col, quantiles in outlier_rules.items():
            if col in df_processed.columns and pd.api.types.is_numeric_dtype(
                df_processed[col]
            ):
                if is_fitting:
                    lower_q_val, upper_q_val = quantiles
                    lower_bound = df_processed[col].quantile(lower_q_val)
                    upper_bound = df_processed[col].quantile(upper_q_val)
                    self._fit_params[f"{col}_lower_bound"] = lower_bound
                    self._fit_params[f"{col}_upper_bound"] = upper_bound
                else:
                    lower_bound = self._fit_params.get(f"{col}_lower_bound")
                    upper_bound = self._fit_params.get(f"{col}_upper_bound")

                if lower_bound is not None and upper_bound is not None:
                    initial_rows = len(df_processed)
                    df_processed = df_processed[
                        (df_processed[col] >= lower_bound)
                        & (df_processed[col] <= upper_bound)
                    ]
                    rows_removed = initial_rows - len(df_processed)
                    if rows_removed > 0:
                        logger.info(
                            f"Filtered {rows_removed} rows from column '{col}' using bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
                        )
                else:
                    logger.warning(
                        f"Outlier bounds for '{col}' not found or not learned. Skipping outlier removal for this column."
                    )
            elif col not in df_processed.columns:
                logger.warning(f"Outlier removal: Column '{col}' not found.")
            else:  # Not numeric
                logger.warning(
                    f"Outlier removal: Column '{col}' is not numeric. Skipping."
                )
        return df_processed

    def _define_features_to_use(self, df: pd.DataFrame) -> List[str]:
        """Defines the list of features to be used by the model based on availability after processing."""
        # This list should ideally be more configurable or derived from feature selection.
        # For now, it's based on legacy + engineered features.
        candidate_features = []
        base_temp_cols = self.feature_eng_config.get("base_temp_cols", [])
        for temp_base in base_temp_cols:
            candidate_features.extend(
                [
                    f"{temp_base}_mean",
                    f"{temp_base}_std",
                    f"{temp_base}_min",
                    f"{temp_base}_max",
                    f"{temp_base}_range",
                ]
            )

        transmission_col = self.feature_eng_config.get("transmission_col")
        if transmission_col:
            candidate_features.extend(
                [
                    f"{transmission_col}_high_signal",
                    f"{transmission_col}_mean_by_sensor",
                    f"{transmission_col}_std_by_sensor",
                    f"{transmission_col}_deviation_from_sensor_mean",
                ]
            )

        # Add original numeric columns that are not IDs, target, or base for engineered features
        # if they are specified in a config or by a feature selection strategy.
        # For now, focus on the engineered ones from legacy logic.

        available_features = [
            f
            for f in candidate_features
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
        ]

        if not available_features:
            logger.warning(
                "No candidate engineered features were found/numeric in the DataFrame. "
                "Consider revising feature engineering or configuration."
            )
            # Fallback: use all numeric columns not ID or target (dangerous without control)
            # For safety, returning empty if no specified features are found and numeric.
            logger.error(
                "No usable features defined or found. Aborting feature selection."
            )
            return []

        logger.info(
            f"Selected features for model based on availability and numeric type: {available_features}"
        )
        return available_features

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Fits the preprocessor on the data and transforms it.
        This includes cleaning, feature engineering, and outlier removal.
        Learned parameters are stored for use by the `transform` method.

        Args:
            df (pd.DataFrame): The raw input DataFrame.

        Returns:
            Tuple[pd.DataFrame, pd.Series, List[str]]:
                - Processed DataFrame of features (X).
                - Series for the target variable (y).
                - List of feature names used.
        """
        self._fit_params = {}  # Reset fit parameters for a new fit
        logger.info("Starting preprocessing fit_transform...")

        df_processed = df.copy()

        # 1. Initial FillNA for columns used directly in feature engineering (e.g., base_temp_cols)
        cols_for_initial_fill = list(
            set(
                self.feature_eng_config.get("base_temp_cols", [])
                + [self.feature_eng_config.get("transmission_col")]
            )
        )
        cols_for_initial_fill = [
            c for c in cols_for_initial_fill if c and c in df_processed.columns
        ]  # Filter None and non-existent
        if cols_for_initial_fill:
            df_processed = self._fillna(
                df_processed, cols_for_initial_fill, is_fitting=True
            )

        # 2. Feature Engineering
        df_processed = self._engineer_features(df_processed, is_fitting=True)

        # 3. Outlier Removal (can be on raw or engineered features, as per config)
        df_processed = self._remove_outliers(
            df_processed, is_fitting=True
        )  # This step might remove rows

        # 4. Define final feature list from available columns
        self.base_features = self._define_features_to_use(df_processed)
        if not self.base_features:
            logger.error(
                "No features selected after engineering and definition. Aborting."
            )
            return (
                pd.DataFrame(),
                pd.Series(dtype="object"),
                [],
            )  # Return empty if no features

        # 5. Final FillNA for all selected features (to ensure no NaNs go to model)
        # This handles NaNs that might have been introduced by merges with missing group keys, etc.
        df_processed = self._fillna(df_processed, self.base_features, is_fitting=True)

        # 6. Separate features and target
        if self.target_column not in df_processed.columns:
            logger.error(
                f"Target column '{self.target_column}' not found in processed data."
            )
            raise ValueError(f"Target column '{self.target_column}' not found.")

        X = df_processed[self.base_features].copy()
        y = df_processed[self.target_column].copy()

        # Final check for NaNs in X and y (should be handled, but as a safeguard)
        if X.isnull().sum().any():
            logger.warning(
                f"NaNs still present in features after all preprocessing. Columns: {X.isnull().sum()[X.isnull().sum() > 0].index.tolist()}"
            )
            logger.info("Attempting final fill with 0 for remaining NaNs in features.")
            X = X.fillna(0)  # Last resort fill

        if y.isnull().any():
            logger.warning(
                f"Target column '{self.target_column}' contains {y.isnull().sum()} NaN values. Removing these rows and corresponding features."
            )
            valid_y_indices = ~y.isnull()
            y = y[valid_y_indices]
            X = X[valid_y_indices]
            if X.empty:
                logger.error(
                    "No data left after removing NaNs from target. Check data quality or preprocessing steps."
                )
                raise ValueError("No data left after removing NaNs from target.")

        logger.info(
            f"Preprocessing fit_transform finished. Feature shape: {X.shape}, Target shape: {y.shape}"
        )
        logger.info(f"Features used: {self.base_features}")
        return X, y, self.base_features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new data using parameters learned during `fit_transform`.

        Args:
            df (pd.DataFrame): The raw input DataFrame (features only, target should not be present or will be ignored).

        Returns:
            pd.DataFrame: Processed DataFrame of features.
        """
        logger.info("Starting preprocessing transform...")
        if not self._fit_params:
            logger.error("Preprocessor has not been fitted. Call fit_transform first.")
            raise RuntimeError("Preprocessor must be fitted before calling transform.")
        if not self.base_features:
            logger.error("Base features not defined after fitting. Cannot transform.")
            raise RuntimeError(
                "Base features not defined in preprocessor after fitting."
            )

        df_transformed = df.copy()

        # 1. Initial FillNA
        cols_for_initial_fill = list(
            set(
                self.feature_eng_config.get("base_temp_cols", [])
                + [self.feature_eng_config.get("transmission_col")]
            )
        )
        cols_for_initial_fill = [
            c for c in cols_for_initial_fill if c and c in df_transformed.columns
        ]
        if cols_for_initial_fill:
            df_transformed = self._fillna(
                df_transformed, cols_for_initial_fill, is_fitting=False
            )

        # 2. Feature Engineering (using stored grouped_stats from _fit_params)
        df_transformed = self._engineer_features(df_transformed, is_fitting=False)

        # 3. Outlier Clipping (not removal, to keep row count consistent for transform)
        # Note: The _remove_outliers method currently filters rows. For transform, usually clipping is preferred.
        # Modifying this for transform: apply clipping instead of filtering.
        outlier_rules = self.config.get("outlier_quantiles", {})
        for col in outlier_rules.keys():
            if col in df_transformed.columns and pd.api.types.is_numeric_dtype(
                df_transformed[col]
            ):
                lower_bound = self._fit_params.get(f"{col}_lower_bound")
                upper_bound = self._fit_params.get(f"{col}_upper_bound")
                if lower_bound is not None and upper_bound is not None:
                    df_transformed[col] = df_transformed[col].clip(
                        lower_bound, upper_bound
                    )
                else:
                    logger.warning(
                        f"Outlier bounds for '{col}' not found in fit_params. Skipping clipping for this column in transform."
                    )

        # 4. Select and Reorder features to match `base_features`
        # Ensure all base_features are present, fill if missing (e.g. if an engineered feature failed for some rows)
        missing_features_for_transform = [
            f for f in self.base_features if f not in df_transformed.columns
        ]
        if missing_features_for_transform:
            logger.warning(
                f"During transform, some base features are missing: {missing_features_for_transform}. They will be created as NaN columns before fillna."
            )
            for f in missing_features_for_transform:
                df_transformed[f] = pd.NA  # Or np.nan

        X_transformed = df_transformed[self.base_features].copy()  # Select and order

        # 5. Final FillNA for selected features
        X_transformed = self._fillna(
            X_transformed, self.base_features, is_fitting=False
        )

        # Final check for NaNs and ensure correct column order
        if X_transformed.isnull().sum().any():
            logger.warning(
                f"NaNs present in transformed features after fillna. Columns: {X_transformed.isnull().sum()[X_transformed.isnull().sum() > 0].index.tolist()}"
            )
            logger.info(
                "Attempting final fill with 0 for remaining NaNs in transformed features."
            )
            X_transformed = X_transformed.fillna(0)

        logger.info(
            f"Preprocessing transform finished. Feature shape: {X_transformed.shape}"
        )
        return X_transformed[self.base_features]  # Ensure final order

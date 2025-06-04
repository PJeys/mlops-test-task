import os
import pickle
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Global variables (bad practice!)
MODEL_PATH = "/tmp/model.pkl"
DATA_PATH = "mlops/beehero-mlops-assignment/data/colony_size.csv"
FEATURES = None  # Will be set later


def load_and_process_data():
    """Load all the data and do preprocessing"""
    global FEATURES

    print("Loading data...")
    # Load main sensor data
    try:
        data = pd.read_csv(DATA_PATH)
    except:
        print("Error loading data! Using backup...")
        data = pd.read_csv("backup_" + DATA_PATH)

    # Hardcoded sensor columns
    temp_sensors = ["temperature_sensor", "temperature_gateway"]

    # Fill missing values with mean (not always the best approach)
    for col in temp_sensors:
        if col in data.columns:
            mean_val = data[col].mean()
            data[col] = data[col].fillna(mean_val)

    # Create features
    print("Engineering features...")

    # Temperature features
    # Calculate temperature statistics per sensor_id
    temp_stats = data.groupby("sensor_id")[temp_sensors].agg(
        ["mean", "std", "min", "max"]
    )
    temp_stats.columns = ["_".join(col).strip() for col in temp_stats.columns.values]
    data = data.merge(temp_stats, on="sensor_id")

    # Create aggregate temperature features
    data[["sensor_temp_range", "gateway_temp_range"]] = (
        data[[f"{temp}_max" for temp in temp_sensors]].values
        - data[[f"{temp}_min" for temp in temp_sensors]].values
    )

    # Transmission strength features
    if "ihs_to_gw_transmission_strength" in data.columns:
        # Flag high strength transmissions
        data["high_transmission"] = (
            data["ihs_to_gw_transmission_strength"] > 10
        ).astype(int)

        # Create feature for transmission variability by sensor
        transmission_stats = data.groupby("sensor_id")[
            "ihs_to_gw_transmission_strength"
        ].agg(["mean", "std"])
        transmission_stats.columns = [
            "ihs_to_gw_transmission_strength_mean",
            "ihs_to_gw_transmission_strength_std",
        ]
        data = data.merge(transmission_stats, on="sensor_id", suffixes=("", "_avg"))

        # Calculate deviation from average transmission strength
        data["transmission_deviation"] = abs(
            data["ihs_to_gw_transmission_strength"]
            - data["ihs_to_gw_transmission_strength_mean"]
        )

    # Remove outliers (hardcoded thresholds)
    print("Removing outliers...")
    data = data[data["temperature_sensor_mean"] < 50]  # Temps above 50C are errors
    data = data[
        data["temperature_sensor_mean"] > 10
    ]  # Temps below 10C mean dead colony

    FEATURES = [
        "temperature_sensor_mean",
        "temperature_gateway_mean",
        "temperature_sensor_std",
        "temperature_gateway_std",
        "sensor_temp_range",
        "gateway_temp_range",
        "ihs_to_gw_transmission_strength_mean",
        "ihs_to_gw_transmission_strength_std",
        "high_transmission",
        "transmission_deviation",
    ]

    # Only keep features that exist
    FEATURES = [f for f in FEATURES if f in data.columns]

    return data


def train_model(data):
    """Train the random forest model"""
    print("Preparing training data...")

    # Remove any remaining NaN values (shouldn't be any, but just in case)
    data = data.dropna()

    # Prepare features and labels
    feature_cols = [c for c in FEATURES if c in data.columns]

    X = data[feature_cols]
    y = data["size"]

    # Check class distribution
    print("Class distribution:")
    print(y.value_counts())

    # Split data (no stratification - could lead to imbalanced test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model with hardcoded hyperparameters
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")

    # Feature importance (crude printing)
    print("\nFeature Importance:")
    for feat, imp in zip(feature_cols, model.feature_importances_):
        print(f"{feat}: {imp:.3f}")

    return model, test_score


def save_model(model, score):
    """Save the model to disk"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save model (overwrites previous version without backup!)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save some metadata (in a very crude way)
    with open(MODEL_PATH.replace(".pkl", "_info.txt"), "w") as f:
        f.write(f"Model trained on: {pd.Timestamp.now()}\n")
        f.write(f"Test accuracy: {score}\n")
        f.write(f"Features used: {FEATURES}\n")

    print(f"Model saved to {MODEL_PATH}")


def predict_new_data(new_data_path):
    """Make predictions on new data"""
    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load and process new data (copy pasted from above)
    new_data = pd.read_csv(new_data_path)

    temp_sensors = ["temperature_sensor", "temperature_gateway"]

    for col in temp_sensors:
        if col in data.columns:
            mean_val = data[col].mean()
            data[col] = data[col].fillna(mean_val)

    # Temperature features
    # Calculate temperature statistics per sensor_id
    temp_stats = data.groupby("sensor_id")[temp_sensors].agg(
        ["mean", "std", "min", "max"]
    )
    temp_stats.columns = ["_".join(col).strip() for col in temp_stats.columns.values]
    data = data.merge(temp_stats, on="sensor_id")

    # Create aggregate temperature features
    data[["sensor_temp_range", "gateway_temp_range"]] = (
        data[[f"{temp}_max" for temp in temp_sensors]].values
        - data[[f"{temp}_min" for temp in temp_sensors]].values
    )

    # Transmission strength features
    if "ihs_to_gw_transmission_strength" in data.columns:
        # Flag high strength transmissions
        data["high_transmission"] = (
            data["ihs_to_gw_transmission_strength"] > 10
        ).astype(int)

        # Create feature for transmission variability by sensor
        transmission_stats = data.groupby("sensor_id")[
            "ihs_to_gw_transmission_strength"
        ].agg(["mean", "std"])
        transmission_stats.columns = [
            "ihs_to_gw_transmission_strength_mean",
            "ihs_to_gw_transmission_strength_std",
        ]
        data = data.merge(transmission_stats, on="sensor_id", suffixes=("", "_avg"))

        # Calculate deviation from average transmission strength
        data["transmission_deviation"] = abs(
            data["ihs_to_gw_transmission_strength"]
            - data["ihs_to_gw_transmission_strength_mean"]
        )

    # Remove outliers (hardcoded thresholds)
    print("Removing outliers...")
    data = data[data["temperature_sensor_mean"] < 50]  # Temps above 50C are errors
    data = data[
        data["temperature_sensor_mean"] > 10
    ]  # Temps below 10C mean dead colony

    FEATURES = [
        "temperature_sensor_mean",
        "temperature_gateway_mean",
        "temperature_sensor_std",
        "temperature_gateway_std",
        "sensor_temp_range",
        "gateway_temp_range",
        "ihs_to_gw_transmission_strength_mean",
        "ihs_to_gw_transmission_strength_std",
        "high_transmission",
        "transmission_deviation",
    ]

    # Only keep features that exist
    FEATURES = [f for f in FEATURES if f in data.columns]

    return predictions


# Main execution
if __name__ == "__main__":
    data = load_and_process_data()

    model, score = train_model(data)

    save_model(model, score)

    # Quick test prediction
    print("\nMaking test predictions...")
    sample = data.sample(5)
    features = [c for c in FEATURES if c in sample.columns]
    predictions = model.predict(sample[features])
    print("Sample predictions:", predictions)
    print("Actual values:", sample["size"].values)

    print("\nDone!")

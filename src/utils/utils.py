import yaml
import pickle
import json
from pathlib import Path
from typing import Any, Dict, Union
from datetime import datetime


def load_config(config_path: Union[str, Path] = "src/config/config.yaml") -> Dict:
    """Loads YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_pickle(obj: Any, filepath: Union[str, Path]):
    """Saves an object as a pickle file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Loads a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_json(data: Dict, filepath: Union[str, Path]):
    """Saves a dictionary as a JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Union[str, Path]) -> Dict:
    """Loads a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def get_timestamp_version() -> str:
    """Generates a timestamp string for versioning."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

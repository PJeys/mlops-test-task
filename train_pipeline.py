import argparse
import sys
import os
from pathlib import Path

# Ensure src directory is in Python path
# This allows 'from src.train import run_training' when running train_pipeline.py from root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.train import run_training  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402 Logger from src.utils

logger = get_logger("train_pipeline_orchestrator")


def main():
    parser = argparse.ArgumentParser(
        description="Main training pipeline for Colony Strength Classifier."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="src/config/config.yaml",  # Default relative to project root
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    # Resolve config_path to absolute to avoid issues if CWD changes
    config_file_path = Path(args.config_path).resolve()
    if not config_file_path.is_file():
        logger.error(f"Configuration file not found at: {config_file_path}")
        sys.exit(1)

    logger.info("=================================================")
    logger.info("🚀 Starting ML Training Pipeline Orchestration 🚀")
    logger.info(f"Using configuration file: {config_file_path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info("=================================================")

    try:
        run_training(config_path=str(config_file_path))
        logger.info("✅ ML Training Pipeline completed successfully.")
    except Exception as e:
        # run_training already logs traceback, so just log failure here.
        logger.error(
            f"❌ ML Training Pipeline failed with an unhandled exception in orchestrator: {e}"
        )
        sys.exit(1)  # Exit with error code for CI/CD to catch failure
    finally:
        logger.info("=================================================")
        logger.info("🏁 ML Training Pipeline Orchestration Finished 🏁")
        logger.info("=================================================")


if __name__ == "__main__":
    main()

"""Data loading utilities for the customer-segmentation-rfm project."""

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the project configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load the raw transactions CSV as specified in the project config.

    Parameters
    ----------
    config : dict
        Loaded project configuration (see load_config).

    Returns
    -------
    pd.DataFrame
        The raw, unprocessed transactions data.
    """
    raw_dir = Path(config["data"]["raw"])
    filename = config["data"]["raw_filename"]
    filepath = raw_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {filepath}. "
            "Place rfm_customer_segments.csv in data/raw/."
        )

    return pd.read_csv(filepath, encoding="ISO-8859-1")
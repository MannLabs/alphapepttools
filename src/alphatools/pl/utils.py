from pathlib import Path

import yaml


def load_plot_config(config_path: str = "plot_config.yaml") -> dict:
    """Load the plot configuration file"""
    with Path.open(config_path) as file:
        return yaml.safe_load(file)

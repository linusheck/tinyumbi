"""Locate the original, parameter-instantiated UMB paper models."""

from pathlib import Path


def paper_models(directory):
    directory = Path(directory)
    paths = sorted(directory.glob("*/model.prism"))
    if not paths:
        raise ValueError(f"No paper models found at {directory}; expected */model.prism")
    return paths

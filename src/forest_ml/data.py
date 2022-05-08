from pathlib import Path
from typing import Tuple

import click
import pandas as pd


def get_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path, index_col="Id")
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    return features, target

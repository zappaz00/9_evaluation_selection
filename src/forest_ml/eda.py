from pathlib import Path
from pandas_profiling import ProfileReport

import click
import pandas as pd


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/heart.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
def eda(
        dataset_path: Path
) -> None:
    dataset = pd.read_csv(dataset_path)
    profile = ProfileReport(dataset, title="Pandas Profiling Report", explorative=True)
    profile.to_file("eda_report.html")
from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score

from .data import get_dataset
from .pipeline import create_pipeline
from .model_type import ModelType
from .dim_red_type import DimReduceType


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--red-type",
    default=DimReduceType.NONE,
    type=DimReduceType,
    show_default=True,
)
@click.option(
    "--red-comp",
    default=0,
    type=int,
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--model-type",
    default=ModelType.LOGREG,
    type=ModelType,
    show_default=True,
)
@click.option(
    "--hyperparams",
    default={},
    type=dict,
    show_default=True,
)
def train(
        dataset_path: Path,
        save_model_path: Path,
        random_state: int,
        test_split_ratio: float,
        red_type: DimReduceType,
        red_comp: int,
        use_scaler: bool,
        model_type: ModelType,
        hyperparams: dict,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        pipeline = create_pipeline(red_type, red_comp, use_scaler, model_type, hyperparams, random_state)
        pipeline.fit(features_train, target_train)
        accuracy = accuracy_score(target_val, pipeline.predict(features_val))

        all_params = hyperparams
        all_params['model_type'] = model_type
        all_params['use_scaler'] = use_scaler
        all_params['dim_red_type'] = red_type
        all_params['red_comp'] = red_comp

        mlflow.log_params(all_params)
        mlflow.log_metric("accuracy", accuracy)
        click.echo(f"Accuracy: {accuracy}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
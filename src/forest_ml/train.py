from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
import numpy as np
import distutils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate

from .data import get_dataset
from .pipeline import create_pipeline
from .model_type import ModelType
from .dim_red_type import DimReduceType


def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False


def is_bool(str):
    try:
        bool(distutils.util.strtobool(str))
        return True
    except ValueError:
        return False


@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
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
    default='none',
    type=click.Choice(DimReduceType.__members__),
    callback=lambda c, p, v: getattr(DimReduceType, v) if v else None,
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
    default='logreg',
    type=click.Choice(ModelType.__members__),
    callback=lambda c, p, v: getattr(ModelType, v) if v else None,
    show_default=True,
)
@click.argument('hyperparams', nargs=-1, type=click.UNPROCESSED)
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
    mlflow.set_experiment("default")

    with mlflow.start_run():
        # парсим экстра-параметры
        hyperparams_dict = {}
        for hyperparam in hyperparams:
            # конвертация типов
            curr_split = hyperparam.split('=')
            if curr_split[1].isnumeric():
                curr_split[1] = int(curr_split[1])
            elif is_float(curr_split[1]):
                curr_split[1] = float(curr_split[1])
            elif is_bool(curr_split[1]):
                curr_split[1] = bool(curr_split[1])
            hyperparams_dict.update([curr_split])

        if hyperparams_dict.get('c') is not None:
            hyperparams_dict['C'] = hyperparams_dict.pop('c') # единственный параметр в Uppercase для LogReg

        # создаём пайплайн
        pipeline = create_pipeline(red_type, red_comp, use_scaler, model_type, hyperparams_dict, random_state)

        cv_scores = cross_validate(pipeline,
                                    features_train,
                                    target_train,
                                    cv=5,
                                    return_estimator=True,
                                    return_train_score=True,
                                    scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'])

        # считаем метрики
        metrics = {}
        metrics['accuracy'] = np.mean(cv_scores['test_accuracy'])
        metrics['f1'] = np.mean(cv_scores['test_f1_weighted'])
        metrics['precision'] = np.mean(cv_scores['test_precision_weighted'])
        metrics['recall'] = np.mean(cv_scores['test_recall_weighted'])

        # готовим параметры для MLFlow
        all_params = hyperparams_dict
        all_params['model_type'] = model_type
        all_params['use_scaler'] = use_scaler
        all_params['dim_red_type'] = red_type
        all_params['red_comp'] = red_comp

        # записываем в MLFlow и сохраняем модель
        mlflow.log_params(all_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")
        click.echo(f"Accuracy: {metrics['accuracy']}.")
        click.echo(f"F1 Weighted: {metrics['f1']}.")
        click.echo(f"Precision Weighted: {metrics['precision']}.")
        click.echo(f"Recall Weighted: {metrics['recall']}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
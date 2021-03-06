from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import (
    cross_validate,
    GridSearchCV,
    StratifiedKFold,
    RandomizedSearchCV,
)

from .params import parse_hyperparams
from .data import get_dataset
from .pipeline import create_pipeline
from .defs import DimReduceType, ModelType, TuneType


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
    )
)
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
    "--tuning",
    default="auto_random",
    type=click.Choice(["manual", "auto_random", "auto_grid"]),
    callback=lambda c, p, v: getattr(TuneType, v) if v else None,
    show_default=True,
)
@click.option(
    "--model-type",
    default="logreg",
    type=click.Choice(["logreg", "knn", "randomforest"]),
    callback=lambda c, p, v: getattr(ModelType, v) if v else None,
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--red-type",
    default="none",
    type=click.Choice(["none", "pca", "tsvd"]),
    callback=lambda c, p, v: getattr(DimReduceType, v) if v else None,
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.argument("hyperparams", nargs=-1, type=click.UNPROCESSED)
def train(
    dataset_path: Path,
    save_model_path: Path,
    tuning: TuneType,
    model_type: ModelType,
    random_state: int,
    red_type: DimReduceType,
    use_scaler: bool,
    hyperparams: dict[str, str],
) -> None:
    features, target = get_dataset(dataset_path)
    # mlflow.set_experiment("default")

    with mlflow.start_run():
        hyperparams_dict = parse_hyperparams(hyperparams)

        # ?????????????? ????????????????
        pipeline = create_pipeline(
            red_type, use_scaler, model_type, hyperparams_dict, random_state
        )
        metrics_names = [
            "accuracy",
            "f1_weighted",
            "precision_weighted",
            "recall_weighted",
        ]

        all_params = {}
        if tuning == TuneType.manual:
            all_params = hyperparams_dict

            cv_scores = cross_validate(
                pipeline,
                features,
                target,
                cv=5,
                return_estimator=True,
                return_train_score=True,
                scoring=metrics_names,
            )

            # ???????????????? ???? ???????? ???????????? ?????? ?????????????????? ???????????? ?????? ?????????? ???? ????????????
            pipeline.fit(features, target)
            mlflow.sklearn.log_model(pipeline, "model")
            dump(pipeline, save_model_path)

        elif tuning == TuneType.auto_random or tuning == TuneType.auto_grid:
            params_for_search = {}

            for hyperparam_name, hyperparam_space in hyperparams_dict.items():
                if hyperparam_name == "n_components":
                    pipe_param_name = "reductor__" + hyperparam_name
                else:
                    pipe_param_name = "classifier__" + hyperparam_name

                params_for_search[pipe_param_name] = hyperparam_space

            print(params_for_search)

            inner_cv = StratifiedKFold(
                n_splits=3, shuffle=True, random_state=random_state
            )
            outer_cv = StratifiedKFold(
                n_splits=10, shuffle=True, random_state=random_state
            )

            # ?????? ?????????????????????? ?????????? ???????????????? ?????????????? ???????? ??????????????
            if tuning == TuneType.auto_random:
                param_searcher = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=params_for_search,
                    scoring="accuracy",
                    cv=inner_cv,
                )  # refit=True default
            elif tuning == TuneType.auto_grid:
                param_searcher = GridSearchCV(
                    estimator=pipeline,
                    param_grid=params_for_search,
                    scoring="accuracy",
                    cv=inner_cv,
                )  # refit=True default

            # ?? ???????????? ???????????????? ???????????????????? ???????? ?????????? predict,
            # ?????????????? ?????????????????????????? ???????????????????? ?? best_estimator.
            cv_scores = cross_validate(
                param_searcher,
                X=features,
                y=target,
                cv=outer_cv,
                scoring=metrics_names,
                return_estimator=True,
                return_train_score=True,
            )
            best_estimator = cv_scores["estimator"][
                np.argmax(cv_scores["test_accuracy"])
            ]
            best_params = best_estimator.get_params()
            for best_param_name, best_param_val in best_params.items():
                if "classifier__" in best_param_name:
                    all_params[
                        best_param_name.replace("classifier__", "")
                    ] = best_param_val
                elif "reductor__" in best_param_name:
                    all_params[
                        best_param_name.replace("reductor__", "")
                    ] = best_param_val

            # ???????????????? ???? ???????? ???????????? ?????? ?????????????????? ???????????? ?????? ?????????? ???? ????????????
            best_estimator.fit(features, target)
            mlflow.sklearn.log_model(best_estimator, "model")
            dump(best_estimator, save_model_path)

        # ?????????????? ??????????????
        metrics = {}
        for metrics_name in metrics_names:
            cv_name = "test_" + metrics_name
            metrics[metrics_name] = float(np.mean(cv_scores[cv_name]))
            click.echo(f"{metrics_name}: {metrics[metrics_name]}.")

        # ?????????????? ?????????????????? ?????? MLFlow
        all_params["model_type"] = model_type
        all_params["use_scaler"] = use_scaler
        all_params["dim_red_type"] = red_type

        # ???????????????????? ?? MLFlow ?? ?????????????????? ????????????
        mlflow.log_params(all_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")
        click.echo(f"Model is saved to {save_model_path}.")

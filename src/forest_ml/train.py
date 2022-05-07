from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.model_selection import cross_validate, GridSearchCV, KFold, RandomizedSearchCV

from .params import get_params_distr, parse_hyperparams
from .data import get_dataset
from .pipeline import create_pipeline
from .defs import DimReduceType, ModelType, TuneType


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
    "--tuning",
    default='auto_random',
    type=click.Choice(TuneType.__members__),
    callback=lambda c, p, v: getattr(TuneType, v) if v else None,
    show_default=True,
)
@click.option(
    "--model-type",
    default='logreg',
    type=click.Choice(ModelType.__members__),
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
    default='none',
    type=click.Choice(DimReduceType.__members__),
    callback=lambda c, p, v: getattr(DimReduceType, v) if v else None,
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.argument('hyperparams', nargs=-1, type=click.UNPROCESSED)
def train(
        dataset_path: Path,
        save_model_path: Path,
        tuning: TuneType,
        model_type: ModelType,
        random_state: int,
        red_type: DimReduceType,
        use_scaler: bool,
        hyperparams: dict,
) -> None:
    features, target = get_dataset(dataset_path)
    # mlflow.set_experiment("default")

    with mlflow.start_run():
        hyperparams_dict = parse_hyperparams(hyperparams)

        # создаём пайплайн
        pipeline = create_pipeline(red_type, use_scaler, model_type, hyperparams_dict, random_state)
        metrics_names = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']

        if tuning == TuneType.manual:
            cv_scores = cross_validate(pipeline,
                                       features,
                                       target,
                                       cv=5,
                                       return_estimator=True,
                                       return_train_score=True,
                                       scoring=metrics_names)

        elif tuning == TuneType.auto_random or tuning == TuneType.auto_grid:
            params_for_search = {}

            for hyperparam_name, hyperparam_space in hyperparams_dict.items():
                if hyperparam_name == 'n_components':
                    pipe_param_name = 'reductor__' + hyperparam_name
                else:
                    pipe_param_name = 'classifier__' + hyperparam_name

                params_for_search[pipe_param_name] = hyperparam_space

            print(params_for_search)

            inner_cv = KFold(n_splits=3,  shuffle=True, random_state=random_state)
            outer_cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

            # для внутреннего цикла согласно заданию одна метрика
            if tuning == TuneType.auto_random:
                param_searcher = RandomizedSearchCV(estimator=pipeline, param_distributions=params_for_search, scoring='accuracy', cv=inner_cv)
            elif tuning == TuneType.auto_grid:
                param_searcher = GridSearchCV(estimator=pipeline, param_grid=params_for_search, scoring='accuracy', cv=inner_cv)

            # у всяких сёрчеров параметров есть метод predict, который автоматически адресуется к best_estimator.
            cv_scores = cross_validate(param_searcher, X=features, y=target, cv=outer_cv, scoring=metrics_names, return_estimator=True, return_train_score=True)

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
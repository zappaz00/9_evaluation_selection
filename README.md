# Forest ML

## Usage

This package allows you to train model for classifying forest cover type.
1. Clone this repository to your machine
2. Download [Forest dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction), save csv file locally (default path is data/train.csv in repository's root)
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/) are installed on your machine.
4. Install the project dependencies (*run this and following commands in terminal, from a root of cloned repository*):
```console
poetry install --no-dev
```
5. Run MLFlow UI to see the information about experiments you conducted:
```console
poetry run mlflow ui
```

Package has two scripts:
* *eda* - for exploratory data analysis
* *train* - for training ML models

## Eda
**eda** generates HTML-file with statistic description of dataset (using pandas-profiling). Report can be opened from your browser.

Script may be launched from terminal like this:
```console
poetry run eda -d "data/train.csv"
```
Where:
* -d (--dataset-path) is path to .csv data file

To get help use:
```console
poetry run eda --help
```

## Train
**train** works in two modes:
* manual hyperparameters setting
* automatic hyperparameters tuning (with Grid/Random procedures)

Script may be launched from terminal like this:
### For manual setting: 
```console
poetry run train -d "data/train.csv" -s "data/model.joblib" --tuning 'manual' --model-type 'logreg' --random-state 42 --red-type 'pca' --use-scaler True c=100.0 max_iter=1000 
```

### For Random tuning:
```console
poetry run train -d "data/train.csv" --tuning 'auto_random' --model-type 'knn' --red-type 'pca' --use-scaler True n_neighbors='uniform(1,20)' weights=['uniform','distance'] 
```

### For Grid tuning:
```console
poetry run train -d "data/train.csv" --tuning 'auto_random' --model-type 'knn' --red-type 'pca' --use-scaler True n_neighbors='linspace(1,20)' weights=['uniform','distance'] 
```

Where:
* -d (--dataset-path) is a path to .csv data file. *Default*: data/train.csv
* -s (--save-model-path) is a path where model will store after training. *Default*: data/model.joblib
* --tuning is a choice of hyperparameters tuning type (manual|auto_random|auto_grid). *Default*: auto_random
* --model-type is a choice of model type (logreg|knn|randomforest). *Default*: logreg
* --random-state is a seed for reproducibility of results. *Default*: 42
* --red-type is a choice of reduction dimensionality algorithm (none|pca|tsvd). *Default*: none
* --use-scaler is a flag that tells to pipeline use standard scaler for data or not. *Default*: true
* other args is hyperparams for pipeline steps and must be in form param1=val1 param2=val2 and etc. Value of parameter can be any python expression.

To get help use:
```console
poetry run train --help
```

## Develop guide
Package uses [poetry](https://python-poetry.org/) for controlling dependencies. Please install it before launch and test.
After install poetry and activate virtual environment run from terminal
```console
poetry install
```
And poetry automatically will download and install dependencies. After you can use developer instruments, e.g. pytest:
```console
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/):
```console
nox [-r]
```
![screenshots/nox.PNG](screenshots/nox.PNG)

## Tests
All tests passed:  
![screenshots/tests.PNG](screenshots/tests.PNG)

## Format and lint
This package checked with [black](https://github.com/psf/black) and [flake8](https://pypi.org/project/flake8/)
### Black:
![screenshots/black_format.PNG](screenshots/black_format.PNG)

### Flake8:
![screenshots/flake8_lint.PNG](screenshots/flake8_lint.PNG)

## Type annotation 
Types checked with [mypy](https://mypy.readthedocs.io/en/stable/):  
![screenshots/mypy_check.PNG](screenshots/mypy_check.PNG)

##

## Experiment with manual hyperparameters tuning and CV
For estimating model performance chosen 4 metrics: 
* accuracy, 
* F1 weighted, 
* precision weighted, 
* recall weighted

Below table with results of experiment sorted by accuracy.  

![screenshots/mlflow_manual_tune.png](screenshots/mlflow_manual_tune.png)
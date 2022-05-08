import numpy as np
from click.testing import CliRunner
from faker import Faker
from os.path import exists
import pytest
import os
import pandas as pd

from forest_ml.train import train

csv_path = 'data/forest_data.csv'
model_path = 'data/model.joblib'


def generate_forest_dataset():
    columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
               'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
               'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2',
               'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',
               'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
               'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17',
               'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
               'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29',
               'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
               'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type']

    rows_num = 100
    fake = Faker()
    data = np.zeros((rows_num, len(columns)))

    for row_ctr in range(rows_num):
        Wilderness_Area_pos = fake.pyint(min_value=0, max_value=3)
        Soil_Type_pos = fake.pyint(min_value=0, max_value=39)
        Cover_Type = fake.pyint(min_value=1, max_value=7)

        Wilderness_Area = np.zeros(4, dtype=np.int32)
        Wilderness_Area[Wilderness_Area_pos] = 1

        Soil_Type = np.zeros(40, dtype=np.int32)
        Soil_Type[Soil_Type_pos] = 1

        data_row = np.zeros(10, dtype=np.int32)
        for i in range(10):
            data_row[i] = fake.pyint(min_value=0, max_value=3000)

        data[row_ctr, :] = np.hstack((data_row, Wilderness_Area, Soil_Type, Cover_Type))

    data_indx = pd.Index(np.linspace(0, rows_num, rows_num, endpoint=False), name='Id', dtype=np.int32)
    data_pd = pd.DataFrame(data=data, columns=columns, index=data_indx, dtype=np.int32)

    return data_pd


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


# ---------------Tuning---------------
def test_error_for_invalid_tuning(
        runner: CliRunner
) -> None:
    """It fails when tuning is not manual|auto_random|auto_grid."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv('./forest_tmp_data.csv')

        result = runner.invoke(
            train,
            [
                "-d",
                './forest_tmp_data.csv',
                "--tuning",
                'test',
            ],
        )
        assert result.exit_code != 0
        assert "Invalid value for '--tuning'" in result.output


def test_success_for_manual_tuning(
        runner: CliRunner
) -> None:
    """It success when tuning is manual|auto_random|auto_grid."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--tuning",
                'manual',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


def test_success_for_auto_random_tuning(
        runner: CliRunner
) -> None:
    """It success when tuning is manual|auto_random|auto_grid."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--tuning",
                'auto_random',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


def test_success_for_auto_grid_tuning(
        runner: CliRunner
) -> None:
    """It success when tuning is manual|auto_random|auto_grid."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--tuning",
                'auto_grid',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


# ---------------Model type---------------
def test_error_for_invalid_model_type(
        runner: CliRunner
) -> None:
    """It fails when model-type is not logreg|knn|random_forest."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--model-type",
                'test',
            ],
        )
        assert result.exit_code != 0
        assert "Invalid value for '--model-type'" in result.output


def test_success_for_logreg_model_type(
        runner: CliRunner
) -> None:
    """It success when model-type is logreg|knn|random_forest."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--model-type",
                'logreg',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


def test_success_for_knn_model_type(
        runner: CliRunner
) -> None:
    """It success when model-type is logreg|knn|random_forest."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--model-type",
                'knn',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


def test_success_for_randomforest_model_type(
        runner: CliRunner
) -> None:
    """It success when model-type is logreg|knn|random_forest."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--model-type",
                'randomforest',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


# ---------------Reduction type---------------
def test_error_for_invalid_red_type(
        runner: CliRunner
) -> None:
    """It fails when red-type is not none|pca|tsvd."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--red-type",
                'test',
            ],
        )
        assert result.exit_code != 0
        assert "Invalid value for '--red-type'" in result.output


def test_success_for_none_red_type(
        runner: CliRunner
) -> None:
    """It success when red-type is none|pca|tsvd."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--red-type",
                'none',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


def test_success_for_pca_red_type(
        runner: CliRunner
) -> None:
    """It success when red-type is none|pca|tsvd."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--red-type",
                'pca',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


# ---------------Scaler option---------------
def test_error_for_invalid_scaler_type(
        runner: CliRunner
) -> None:
    """It fails when use-scaler is not True|False."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--use-scaler",
                'test',
            ],
        )
        assert result.exit_code != 0
        assert "Invalid value for '--use-scaler'" in result.output


def test_success_for_true_scaler_type(
        runner: CliRunner
) -> None:
    """It success when use-scaler is True|False."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--use-scaler",
                True,
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


def test_success_for_false_scaler_type(
        runner: CliRunner
) -> None:
    """It success when use-scaler is True|False."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--use-scaler",
                False,
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


# ---------------HyperParams---------------
def test_error_for_invalid_hyperparams(
        runner: CliRunner
) -> None:
    """It fails when spaces presence in wrong places."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--use-scaler",
                True,
                "--model-type",
                'logreg',
                "--tuning",
                'manual',
                'c = 10'
            ],
        )
        assert result.exit_code != 0


def test_success_for_valid_hyperparams(
        runner: CliRunner
) -> None:
    """It success when spaces is delimeters between args."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--use-scaler",
                True,
                "--model-type",
                'logreg',
                "--tuning",
                'manual',
                'c=10'
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


def test_success_for_random_search(
        runner: CliRunner
) -> None:
    """It success when hyperparams defined correctly."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--use-scaler",
                True,
                "--model-type",
                'logreg',
                "--tuning",
                'auto_random',
                'c=loguniform(1.0e-5, 1.0e+5)',
                'max_iter=loguniform(100, 10000)'
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True


def test_success_for_grid_search(
        runner: CliRunner
) -> None:
    """It success when hyperparams defined correctly."""
    with runner.isolated_filesystem():
        os.mkdir('data')

        data = generate_forest_dataset()
        data.to_csv(csv_path)

        result = runner.invoke(
            train,
            [
                "-d",
                csv_path,
                "--use-scaler",
                True,
                "--model-type",
                'knn',
                "--tuning",
                'auto_grid',
                'n_neighbors=linspace(1,100,10,dtype=np.int32)',
                'weights=["uniform","distance"]'
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output
        assert exists(model_path) is True

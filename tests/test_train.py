import numpy as np
from click.testing import CliRunner
from faker import Faker
import pytest
import pandas as pd

from forest_ml.train import train


def generate_forest_dataset():
    columns = ['Id','Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology',
               'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon',
               'Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2',
               'Wilderness_Area3','Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4',
               'Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11',
               'Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17',
               'Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23',
               'Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29',
               'Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35',
               'Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40','Cover_Type']

    rows_num = 15000
    fake = Faker()
    data = np.zeros((rows_num, len(columns)))

    for row_ctr in range(rows_num):
        Wilderness_Area_pos = fake.pyint(min_value=0, max_value=3)
        Soil_Type_pos = fake.pyint(min_value=0, max_value=39)
        Cover_Type = fake.pyint(min_value=1, max_value=7)

        Wilderness_Area = np.zeros(4)
        Wilderness_Area[Wilderness_Area_pos] = 1

        Soil_Type = np.zeros(40)
        Soil_Type[Soil_Type_pos] = 1

        data_row = np.array([row_ctr, fake.pyint(), fake.pyint(), fake.pyint(),
                             fake.pyint(),fake.pyint(),fake.pyint(),fake.pyint(),
                             fake.pyint(),fake.pyint(),fake.pyint()]).T

        data[row_ctr, :] = np.hstack((data_row, Wilderness_Area, Soil_Type, Cover_Type))

    data_pd = pd.DataFrame(data=data, columns=columns)

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
        result = runner.invoke(
            train,
            [
                "--tuning",
                'test',
            ],
        )
        assert result.exit_code == 2
        assert "Invalid value for '--tuning'" in result.output


def test_success_for_manual_tuning(
            runner: CliRunner
    ) -> None:
        """It success when tuning is manual|auto_random|auto_grid."""
        with runner.isolated_filesystem():
            data = generate_forest_dataset()
            data.to_csv('forest_tmp_data.csv')

            result = runner.invoke(
                train,
                [
                    "-d",
                    'forest_tmp_data.csv',
                    "--tuning",
                    'manual',
                ],
            )
            assert result.exit_code == 0
            assert "accuracy" in result.output


def test_success_for_auto_random_tuning(
        runner: CliRunner
) -> None:
    """It success when tuning is manual|auto_random|auto_grid."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--tuning",
                'auto_random',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output


def test_success_for_auto_grid_tuning(
        runner: CliRunner
) -> None:
    """It success when tuning is manual|auto_random|auto_grid."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--tuning",
                'auto_grid',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output


# ---------------Model type---------------
def test_error_for_invalid_model_type(
    runner: CliRunner
) -> None:
    """It fails when model-type is not logreg|knn|random_forest."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--model-type",
                'test',
            ],
        )
        assert result.exit_code == 2
        assert "Invalid value for '--model-type'" in result.output


def test_success_for_logreg_model_type(
            runner: CliRunner
    ) -> None:
        """It success when model-type is logreg|knn|random_forest."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                train,
                [
                    "--model-type",
                    'logreg',
                ],
            )
            assert result.exit_code == 0
            assert "accuracy" in result.output


def test_success_for_knn_model_type(
        runner: CliRunner
) -> None:
    """It success when model-type is logreg|knn|random_forest."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--model-type",
                'knn',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output


def test_success_for_randomforest_model_type(
        runner: CliRunner
) -> None:
    """It success when model-type is logreg|knn|random_forest."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--model-type",
                'randomforest',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output


# ---------------Reduction type---------------
def test_error_for_invalid_red_type(
    runner: CliRunner
) -> None:
    """It fails when red-type is not none|pca|tsvd."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--red-type",
                'test',
            ],
        )
        assert result.exit_code == 2
        assert "Invalid value for '--red-type'" in result.output


def test_success_for_none_red_type(
            runner: CliRunner
    ) -> None:
        """It success when red-type is none|pca|tsvd."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                train,
                [
                    "--red-type",
                    'none',
                ],
            )
            assert result.exit_code == 0
            assert "accuracy" in result.output


def test_success_for_pca_red_type(
        runner: CliRunner
) -> None:
    """It success when red-type is none|pca|tsvd."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--red-type",
                'pca',
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output


# ---------------Scaler option---------------
def test_error_for_invalid_scaler_type(
    runner: CliRunner
) -> None:
    """It fails when use-scaler is not True|False."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--use-scaler",
                'test',
            ],
        )
        assert result.exit_code == 2
        assert "Invalid value for '--use-scaler'" in result.output


def test_success_for_true_scaler_type(
            runner: CliRunner
    ) -> None:
        """It success when use-scaler is True|False."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                train,
                [
                    "--use-scaler",
                    True,
                ],
            )
            assert result.exit_code == 0
            assert "accuracy" in result.output


def test_success_for_false_scaler_type(
        runner: CliRunner
) -> None:
    """It success when use-scaler is True|False."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--use-scaler",
                False,
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output


# ---------------HyperParams---------------
def test_error_for_invalid_hyperparams(
    runner: CliRunner
) -> None:
    """It fails when spaces presence in wrong places."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--use-scaler",
                True,
                "--model-type",
                'logreg',
                "--tuning",
                'manual',
                'c = 10'
            ],
        )
        assert result.exit_code == 1


def test_success_for_valid_hyperparams(
            runner: CliRunner
    ) -> None:
        """It success when spaces is delimeters between args."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                train,
                [
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


def test_success_for_random_search(
        runner: CliRunner
) -> None:
    """It success when hyperparams defined correctly."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
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


def test_success_for_grid_search(
        runner: CliRunner
) -> None:
    """It success when hyperparams defined correctly."""
    with runner.isolated_filesystem():
        result = runner.invoke(
            train,
            [
                "--use-scaler",
                True,
                "--model-type",
                'knn',
                "--tuning",
                'auto_grid',
                'n_neighbors=linspace(1,100,10)',
                'weights=["uniform","distance"]'
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output


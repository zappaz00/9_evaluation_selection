from click.testing import CliRunner
import pytest

from forest_ml.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


# ---------------Tuning---------------
def test_error_for_invalid_tuning(
    runner: CliRunner
) -> None:
    """It fails when tuning is not manual|auto_random|auto_grid."""
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
        result = runner.invoke(
            train,
            [
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
    result = runner.invoke(
        train,
        [
            "--use-scaler",
            True,
            "--model-type",
            'logreg',
            'c = 0'
        ],
    )
    assert result.exit_code == 1


def test_success_for_valid_hyperparams(
            runner: CliRunner
    ) -> None:
        """It success when spaces is delimeters between args."""
        result = runner.invoke(
            train,
            [
                "--use-scaler",
                True,
                "--model-type",
                'logreg',
                'c=0'
            ],
        )
        assert result.exit_code == 0
        assert "accuracy" in result.output


def test_success_for_random_search(
        runner: CliRunner
) -> None:
    """It success when hyperparams defined correctly."""
    result = runner.invoke(
        train,
        [
            "--use-scaler",
            True,
            "--model-type",
            'logreg',
            "--tuning"
            'auto_random'
            'c=loguniform(1.0e-5, 1.0e+5)'
            'max_iter=loguniform(100, 10000)'
        ],
    )
    assert result.exit_code == 0
    assert "accuracy" in result.output


def test_success_for_grid_search(
        runner: CliRunner
) -> None:
    """It success when hyperparams defined correctly."""
    result = runner.invoke(
        train,
        [
            "--use-scaler",
            True,
            "--model-type",
            'knn',
            "--tuning"
            'auto_grid'
            'n_neighbors=linspace(1,100,10)'
            'weights=["uniform","distance"]'
        ],
    )
    assert result.exit_code == 0
    assert "accuracy" in result.output


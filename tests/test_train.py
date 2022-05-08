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


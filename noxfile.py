"""Nox sessions."""

import tempfile
import os
from typing import Any

import nox
from nox.sessions import Session


nox.options.sessions = "black", "flake8", "mypy", "tests"
locations = "src", "noxfile.py"


def install_with_constraints(
    session: Session, *args: str, **kwargs: Any
) -> None:  # noqa: E501
    """Install packages constrained by Poetry's lock file.
    By default newest versions of packages are installed,
    but we use versions from poetry.lock instead to
    guarantee reproducibility of sessions.
    """
    with tempfile.NamedTemporaryFile(delete=False) as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)
        requirements.close()
        os.unlink(requirements.name)


@nox.session(python="3.9")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python="3.9")
def flake8(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "flake8")
    session.run("flake8", *args)


@nox.session(python="3.9")
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run("mypy", *args)


@nox.session(python=False)
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs
    session.run("poetry", "shell")
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "pytest", "Faker")
    session.run("pytest", *args)

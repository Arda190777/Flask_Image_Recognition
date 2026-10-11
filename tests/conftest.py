"""Shared pytest fixtures for the Flask application."""

import pytest

from app import app as flask_app


@pytest.fixture(name="test_app")
def app_fixture():
    """Create a Flask application configured for testing."""
    flask_app.config.update(TESTING=True)
    yield flask_app


@pytest.fixture(name="client")
def client_fixture(test_app):
    """Create a Flask test client."""
    return test_app.test_client()

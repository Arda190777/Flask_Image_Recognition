"""Tests for the Flask application routes."""

from io import BytesIO

import app as app_module


def test_home_page_is_accessible(client):
    """The home page should return a successful response."""
    response = client.get("/")

    assert response.status_code == 200
    assert b"Hand Sign Digit Language Detection" in response.data


def test_valid_upload_returns_prediction(client, monkeypatch):
    """A valid upload should display the predicted digit."""
    monkeypatch.setattr(
        app_module,
        "preprocess_img",
        lambda image_stream: "processed-image",
    )
    monkeypatch.setattr(
        app_module,
        "predict_result",
        lambda processed_image: 7,
    )

    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b"valid-image"), "digit.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"7" in response.data


def test_missing_file_returns_error(client):
    """A request without an uploaded file should return an error."""
    response = client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"File cannot be processed." in response.data


def test_invalid_file_returns_error(client):
    """Invalid image content should be rejected."""
    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b"not-an-image"), "invalid.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"File cannot be processed." in response.data

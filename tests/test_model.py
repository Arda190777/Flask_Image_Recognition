"""Tests for image preprocessing and model prediction."""

from io import BytesIO
from unittest.mock import Mock

import numpy as np
from PIL import Image

import model as model_module
from model import predict_result, preprocess_img


def create_image_stream(mode):
    """Create an in-memory PNG image using the requested colour mode."""
    image_stream = BytesIO()
    image = Image.new(mode, (50, 50), color=0)
    image.save(image_stream, format="PNG")
    image_stream.seek(0)
    return image_stream


def test_preprocess_rgba_image():
    """RGBA input should be converted to the model's RGB input shape."""
    image_stream = create_image_stream("RGBA")

    result = preprocess_img(image_stream)

    assert result.shape == (1, 224, 224, 3)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_grayscale_image():
    """Grayscale input should also be converted into three RGB channels."""
    image_stream = create_image_stream("L")

    result = preprocess_img(image_stream)

    assert result.shape == (1, 224, 224, 3)


def test_predict_result_returns_highest_scoring_class(monkeypatch):
    """Prediction should return the class with the highest model score."""
    fake_model = Mock()
    fake_model.predict.return_value = np.array([[0.10, 0.20, 0.70]])
    monkeypatch.setattr(model_module, "MODEL", fake_model)

    processed_image = np.zeros((1, 224, 224, 3))
    result = predict_result(processed_image)

    assert result == 2

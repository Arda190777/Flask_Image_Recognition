"""Image preprocessing and prediction utilities."""

from pathlib import Path

from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image


MODEL_PATH = Path(__file__).resolve().parent / "digit_model.h5"
MODEL = load_model(str(MODEL_PATH))


def preprocess_img(image_source):
    """Convert an uploaded image into the format expected by the model."""
    with Image.open(image_source) as original_image:
        resized_image = original_image.convert("RGB").resize((224, 224))
        image_array = img_to_array(resized_image) / 255.0

    return np.expand_dims(image_array, axis=0)


def predict_result(processed_image):
    """Return the digit class with the highest prediction score."""
    prediction = MODEL.predict(processed_image, verbose=0)
    return np.argmax(prediction[0], axis=-1)

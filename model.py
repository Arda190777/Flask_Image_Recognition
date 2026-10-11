# Importing required libs
from pathlib import Path

from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Build a portable path to the model stored in the project directory.
MODEL_PATH = Path(__file__).resolve().parent / "digit_model.h5"

# Loading model
model = load_model(str(MODEL_PATH))

def preprocess_img(img_path):
    op_img = Image.open(img_path).convert("RGB")
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


# Predicting function
def predict_result(predict):
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)

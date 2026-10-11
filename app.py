"""Flask application for hand-sign digit recognition."""

from flask import Flask, render_template, request

from model import predict_result, preprocess_img


app = Flask(__name__)


@app.route("/")
def main():
    """Render the image-upload home page."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Process an uploaded image and display the predicted digit."""
    try:
        uploaded_file = request.files["file"]
        if not uploaded_file.filename:
            raise ValueError("No file was selected.")
        processed_image = preprocess_img(uploaded_file.stream)
        prediction = predict_result(processed_image)

        return render_template(
            "result.html",
            predictions=str(prediction),
        )

    except (KeyError, OSError, ValueError) as error:
        app.logger.exception("Prediction failed: %s", error)
        message = "File cannot be processed."
        return render_template("result.html", err=message), 400


if __name__ == "__main__":
    app.run(port=9000, debug=True)

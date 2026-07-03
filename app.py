from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = YOLO("best.pt")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Prediction
    results = model.predict(
        source=filepath,
        imgsz = 224,
        verbose=False
    )

    result = results[0]

    predicted_class = model.names[
        result.probs.top1
    ]

    confidence = (
        result.probs.top1conf.item() * 100
    )

    return render_template(
        "result.html",
        prediction=predicted_class,
        confidence=round(confidence, 2),
        image_path=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)
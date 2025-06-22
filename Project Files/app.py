from flask import Flask, request, render_template, redirect
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # Adjust if your model uses different preprocessing
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Construct the full path to the model file
MODEL_PATH = os.path.join(app.root_path, 'Blood Cell.h5')
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{os.path.basename(MODEL_PATH)}' loaded successfully.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

def predict_image_class(image_path, model):
    if model is None:
        raise Exception("Model not loaded. Cannot perform prediction.")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    img_preprocessed = preprocess_input(img_resized.reshape((1, 224, 224, 3)))

    predictions = model.predict(img_preprocessed)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_idx]
    return predicted_class_label, img_rgb

@app.route("/", methods=["GET", "POST"]) # Keep this as is if you prefer one route for both
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            static_dir = os.path.join(app.root_path, 'static')
            os.makedirs(static_dir, exist_ok=True)

            file_path = os.path.join(static_dir, file.filename)
            file.save(file_path)

            try:
                predicted_class_label, img_rgb = predict_image_class(file_path, model)

                _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                img_str = base64.b64encode(img_encoded).decode('utf-8')

                # Pass the predicted_class_label as 'class_label' and the image data with prefix as 'image_path'
                return render_template("result.html", class_label=predicted_class_label, image_path=f"data:image/png;base64,{img_str}")
            except FileNotFoundError as e:
                return f"Error: {e}", 400
            except Exception as e:
                return f"An error occurred during prediction: {e}", 500
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
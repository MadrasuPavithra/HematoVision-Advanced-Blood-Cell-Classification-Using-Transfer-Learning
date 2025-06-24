from flask import Flask, request, render_template, redirect, url_for
import os
from tensorflow.keras.models import load_model
# Assuming your model was trained with MobileNetV2's preprocessing.
# If your model uses a different preprocessing function, adjust this import accordingly.
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
import base64
import logging # Import logging for better error handling

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Construct the full path to the model file.
# If 'Blood Cell2.h5' is in the same directory as this Python script,
# app.root_path correctly points to that directory.
MODEL_PATH = os.path.join(app.root_path, 'Blood Cell2.h5')

# Verify model existence before attempting to load
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file not found at: {MODEL_PATH}")
    # You might want to raise an error or exit if the model is critical
    # For a web app, it's better to show a graceful error page
    raise FileNotFoundError(f"Model file not found. Please ensure 'Blood Cell2.h5' is in the root directory: {app.root_path}")

try:
    model = load_model(MODEL_PATH)
    logging.info(f"Model 'Blood Cell1.h5' loaded successfully from: {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model from {MODEL_PATH}: {e}")
    # Depending on your application's criticality, you might want to exit here
    raise SystemExit(f"Failed to load the model: {e}")


class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']


def predict_image_class(image_path, model_to_use): # Renamed model to model_to_use to avoid shadowing
    """
    Predicts the class of an image.

    Args:
        image_path (str): The path to the image file.
        model_to_use (tf.keras.Model): The loaded Keras model.

    Returns:
        tuple: A tuple containing the predicted class label (str) and
               the RGB image (numpy.ndarray).
    """
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not read image at {image_path}")
        raise FileNotFoundError(f"Image not found or unreadable at {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Preprocess the image for the model
    # The model expects a batch dimension, hence reshape((1, 224, 224, 3))
    img_preprocessed = preprocess_input(img_resized.reshape((1, 224, 224, 3)))

    predictions = model_to_use.predict(img_preprocessed)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_idx]
    return predicted_class_label, img_rgb


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            logging.warning("No file part in the request")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            logging.warning("No selected file")
            return redirect(request.url)
        if file:
            # Ensure the 'static' directory exists for saving uploaded images
            static_dir = os.path.join(app.root_path, 'static')
            os.makedirs(static_dir, exist_ok=True) # exist_ok=True prevents error if dir exists

            file_path = os.path.join(static_dir, file.filename)
            try:
                file.save(file_path)
                logging.info(f"File saved to: {file_path}")
            except Exception as e:
                logging.error(f"Error saving file {file.filename}: {e}")
                return render_template("error.html", message=f"Error saving file: {e}")

            try:
                # Pass the global 'model' object to the prediction function
                predicted_class_label, img_rgb = predict_image_class(file_path, model)

                # Convert image to string for displaying in HTML
                # Use a higher quality image encoding if desired, e.g., '.jpeg' with quality
                _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                img_str = base64.b64encode(img_encoded).decode('utf-8')

                return render_template("result.html", class_label=predicted_class_label, img_data=img_str)
            except FileNotFoundError as e:
                logging.error(f"Prediction error (file not found): {e}")
                return render_template("error.html", message=f"Prediction Error: {e}"), 400
            except Exception as e:
                logging.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
                return render_template("error.html", message=f"An error occurred during prediction: {e}"), 500
            finally:
                # Optional: Clean up the uploaded file after prediction
                # os.remove(file_path)
                # logging.info(f"Cleaned up uploaded file: {file_path}")
                pass

    return render_template("home.html")


if __name__ == "__main__":
    # Ensure templates directory exists for Flask to find HTML files
    # (Though Flask automatically looks for 'templates' in the root)
    # os.makedirs(os.path.join(app.root_path, 'templates'), exist_ok=True)
    app.run(debug=True)

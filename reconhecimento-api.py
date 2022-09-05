import numpy as np
import tensorflow as tf
import os
from flask import Flask, request
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

# Consts
CLASS_NAMES = [
    'cebola', 'creme-de-leite', 'dente-de-alho', 'frango', 'katchup', 'tomate'
]
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 160
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create Flask App
app = Flask(__name__)

# Load model
model = load_model(os.path.join(BASE_DIR, 'model.hdf5'))

# Routes
@app.route('/predictUrl', methods=['POST'])
def predict_url():
    try:
        if request.method == 'POST':
            request_data = request.get_json()
            prediction_result = predict_url(request_data['url'])
    except:
        prediction_result = {
            "ingredient_predicted": "undefined",
            "probability": 0,
            "message": "Ocorreu um erro tentar predizer a imagem da url recebida."
        }
    finally:
        return prediction_result, 200


@app.route('/predict', methods=['POST'])
def predict_file():
    try:
        if 'file' not in request.files:
            return "Please try again. The Image doesn't exist"

        # Get the file from post request
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(BASE_DIR, secure_filename(filename))
        file.save(file_path)

        prediction_result = _predict(file_path)
        # Remove the file after prediction
        os.remove(file_path)
    except:
        prediction_result = {
            "ingredient_predicted": "undefined",
            "probability": 0,
            "message": "Ocorreu um erro tentar predizer a imagem recebido."
        }
    finally:
        return prediction_result, 200


# Methods
def _predict(image_file):
    image = tf.keras.preprocessing.image.load_img(image_file,
                                                  target_size=IMAGE_SIZE)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, 0)

    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    print("Predição: {:2.2f}% | {}".format(100 * np.max(prediction),
                                           CLASS_NAMES[predicted_label]))
    prediction_result = {
        "ingredient_predicted": CLASS_NAMES[predicted_label],
        "probability": 100 * np.max(prediction)
    }
    return prediction_result

def _predict_url(image_origin):
    image_file = tf.keras.utils.get_file(origin=image_origin)
    return _predict(image_file)


# Run
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5500))
    app.run(host='0.0.0.0', port=port, debug=True)
from __future__ import division, print_function
import os
import numpy as np

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


from PIL import Image
import tensorflow as tf


def names(number):
    if number == 0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'


# labels = ['No', 'Yes']
model = tf.keras.models.load_model(
    r"C:\Users\Aryan\FLASK\brain_tumor_detection_model.h5")


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':

        f = request.files['image']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        img = Image.open(file_path)
        img = np.array(img.resize((128, 128)))

        img = img.reshape(1, 128, 128, 3)

        prediction = model.predict_on_batch(img)

        classification = np.where(prediction == np.amax(prediction))[1][0]

    return render_template('pridiction.html', prediction_text=f"{prediction[0][classification]*100:.4f}% Confidence This Is {names(classification)}")


@app.route('/PredictAgain', methods=['POST'])
def PredictAgain():
    '''
    For rendering results on HTML GUI
    '''

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

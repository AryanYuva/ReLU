import cv2
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, url_for, request, app, jsonify
app = Flask(__name__)

from tensorflow.keras.models import load_model

model = load_model("cnnmodel.h5")

labels = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike',
       'person']

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict2",methods=["GET","POST"])
def predict_image():
    file = request.files['image'] 
    filename = file.filename
    if str(filename).strip():
        imgData = cv2.imread(filename)
        rsimg = cv2.resize(imgData, (32, 32))
        pred = np.argmax(model.predict(np.expand_dims(rsimg,axis=0)))
        try:
            print(pred)
            # prediction = str(model.predict(rsimg.reshape(1, -1))[0])
            res = "Prediction : The given image is "+labels[pred]
            return res
        except Exception as e:
            print("can not able to resize the image :(")
            return "can not able to resize the image :("

    else:
        return render_template("home.html")



if __name__ == "__main__":
    app.run(debug=True)

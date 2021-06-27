# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os

print(tf.__version__)

ALLOWED_EXT = set(['png', 'jpg', 'jpeg'])
BASE = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE, 'static/upload')
CLASSES = np.array(['Buildings', 'Forest' ,'Glacier' ,'Mountain' ,'Sea' ,'Street'])

app = Flask(__name__)

app.secret_key = "marj-app"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024

model = None

def get_model():
    global model
    if model == None:
        model = keras.models.load_model(os.path.join(BASE, 'model/weights_vgg16.h5'))
    return model

def allowed_file(filename):
    if '.' in filename and filename.split('.')[-1].lower() in ALLOWED_EXT:
        return True
    return False

def load_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def classify(img_path):
    imgs = load_image(img_path)
    model = get_model()
    result = model.predict(imgs)
    return result

@app.route('/')
def index():
    return render_template('main.html', pred=None, image=None)

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return render_template('main.html', pred="No file part", image=None)
    file = request.files['file']
    if file.filename == '':
        return render_template('main.html', pred="No Image selected form uploading", image=None)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        result = classify(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_url = url_for('static', filename='upload/' + filename)
        return render_template('main.html', pred=str('Class: {}, Confidence: {}'.format(CLASSES[result.argmax()], result.max())), image=image_url)
    else:
        return render_template('main.html', pred="Format not supported", image=None)

if __name__ == '__main__':
    app.run(debug=True)


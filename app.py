import os
from flask import Flask, render_template, request, jsonify
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained CNN model
model = tf.keras.models.load_model('trained_cnn_model.h5')

# Function to preprocess the image before passing it to the model


# def preprocess_image(file_storage):
#     img = Image.open(file_storage)
#     img = img.resize((224, 224))  # Resize the image to the required input size of your model
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
#     return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    # species = None
    # if request.method == 'POST':    
    #     # weight = float(request.form['weight'])
    #     # length1 = float(request.form['length1'])
    #     # length2 = float(request.form['length2'])
    #     # length3 = float(request.form['length3'])
    #     # height = float(request.form['height'])
    #     # width = float(request.form['width'])
    #     image = request.form['plantImage']

    #     # Use the loaded model to make the prediction
    #    # data = [[weight, length1, length2, length3, height, width]\]
    #     species = model.predict(image)[0]

    if request.method == 'POST':
        # Check if the post request has the file part
        if 'plantImage' not in request.files:
            return jsonify({'error': 'No file part'})

        image_file = request.files['plantImage']
        print(image_file)
        # Check if the file is selected and has an allowed extension
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'})

        if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({'error': 'Invalid file format'})

        # Preprocess the image and make predictions
        # img_array = preprocess_image(image_file)
        # print(img_array)

        img = Image.open(image_file)
        img = img.resize((224, 224))  # Resize the image to the required input size of your model
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        print("Predicted class index:", predicted_class_index)

        # predictions = model.predict(img_array)
        # p = np.argmax(predictions)
        # print(p)
         # Replace classes with your specific class labels
        classes = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
        predicted_class = classes[np.argmax(predictions[0])]
        print(predicted_class)
        return render_template('index.html',prediction= predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
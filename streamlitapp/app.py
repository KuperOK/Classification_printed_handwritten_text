import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

custom_objects = {'BatchNormalization': tf.keras.layers.BatchNormalization}
model = tf.keras.models.load_model('CNN_resnet50based_.h5', custom_objects=custom_objects)
# model = tf.keras.models.load_model('model_more_complex_CNN.h5')


def preprocess_image(image, image_size=(320, 157)):
    img = image.convert('RGB').resize(image_size)
    img_array = np.array(img).T / 255.0
    if img_array.shape[0] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array.transpose((1, 2, 0))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


st.title("Handwritten vs Printed Text Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    img_array = preprocess_image(image)

    prediction = model.predict(img_array)
    prediction_probability = prediction[0][0]

    if prediction_probability > 0.5:
        prediction_label = "Printed Text"
    else:
        prediction_label = "Handwritten Text"
    st.write(f"Prediction: {prediction_label}")
    st.write(f"Prediction Probability: {prediction_probability:.2f}")

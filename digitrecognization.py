import pickle
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
loaded_model = tf.keras.models.load_model("digit_model.h5")

# Define function for prediction
def predict_digit(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    input_image_resize = cv2.resize(grayscale, (28, 28))
    input_image_resize = input_image_resize / 255
    image_reshaped = np.reshape(input_image_resize, [1, 28, 28])
    input_prediction = loaded_model.predict(image_reshaped)
    input_pred_label = np.argmax(input_prediction)
    return input_pred_label

# Streamlit app
st.title('Handwritten Digit Recognition')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        prediction = predict_digit(image)
        st.write(f'The Handwritten Digit is recognized as {prediction}')

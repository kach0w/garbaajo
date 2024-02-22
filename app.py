import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
st.title("Garbage Classification App")

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.model_from_json(open("model.json", "r").read())
    model.load_weights("weights.h5")
    return model
def preprocess_image(image):
    # Resize the image to match the input shape of your model
    resized_image = cv2.resize(image, (100, 100))
    # Normalize the image pixel values
    normalized_image = resized_image / 255.0
    # Add batch dimension
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Unable to open camera.")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        st.error("Error reading frame.")
        break

    # Process the captured frame
    processed_frame = preprocess_image(frame)

    # Perform prediction
    prediction = model.predict(processed_frame)

    # Display prediction or any other output
    st.write("Prediction:", prediction)  # Modify this to display your prediction

    # Display the captured frame
    st.image(frame, channels="BGR")

    # Close the camera when the "Close Camera" button is clicked
    if st.button("Close Camera"):
        break

    # Release the camera
cap.release()

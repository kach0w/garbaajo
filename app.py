import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.title("Garbage Classification App")

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.model_from_json(open("model.json", "r").read())
    model.load_weights("weights.h5")
    return model
def preprocess_image(image):
    image = Image.open(image)
    resized_image = image.resize((100, 100))
    # normalized_image = resized_image / 255.0
    preprocessed_image = np.expand_dims(resized_image, axis=0)
    return preprocessed_image

model = load_model()
frame = st.camera_input("Take Your Photo");
if frame is not None:
    processed_frame = preprocess_image(frame)
    prediction = model.predict(processed_frame)
    st.write("Prediction:", prediction) 
    st.image(frame, channels="BGR")
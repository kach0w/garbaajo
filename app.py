import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import torch

st.title("Garbage Classification App")
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.load("model.pth")
    return model
def preprocess_image(image):
    image = Image.open(image)
    resized_image = image.resize((100, 100))
    preprocessed_image = np.expand_dims(resized_image, axis=0)
    return preprocessed_image

model = torch.load("model.pth")
frame = st.camera_input("Take Your Photo");
if frame is not None:
    processed_frame = preprocess_image(frame)
    
    outputs = model(images[0])
    _, predictions = torch.max(outputs.data, 1)
    st.write(predictions)
    # arr = ["a battery", "biological stuff", "trash", "things to be recycled"]
    # c = arr[np.argmax(prediction)]

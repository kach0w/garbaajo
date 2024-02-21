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

def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((100, 100))
    img_array = np.array(img)
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

model = load_model()

picture = st.camera_input("Take a picture")
if picture is not None:
    img = load_image(picture)
    pred = model.predict(img)
    num = np.argmax(pred)    
    pred_arr = ["Other", "Organic", "Trash", "Recycling"]
    num = pred_arr[num]
    col1, col2 = st.columns(2)
    with col1:
        st.image(picture, caption="Uploaded Image", width=350)
    with col2:
        col2.subheader(str(num))
        st.write(str(pred))

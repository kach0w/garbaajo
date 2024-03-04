import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import torch

st.title("Garbage Classification App")
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 100 - (5-1) /2 => 48
        self.conv2 = nn.Conv2d(6, 8, 5) # 48 - 4 / 2 = 22
        self.fc1 = nn.Linear(8*22*22, 12)
        # self.fc2 = nn.Linear(12, 12) 

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, 8*22*22)
        x = self.fc1(x)
        # x = self.fc2(x)
        return x
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.load("model.pth")
    return model
def preprocess_image(image):
    image = Image.open(image)
    resized_image = image.resize((100, 100))
    preprocessed_image = np.expand_dims(resized_image, axis=0)
    return preprocessed_image

model = load_model()
frame = st.camera_input("Take Your Photo");
if frame is not None:
    processed_frame = preprocess_image(frame)
    
    outputs = model(images[0])
    _, predictions = torch.max(outputs.data, 1)
    st.write(predictions)
    # arr = ["a battery", "biological stuff", "trash", "things to be recycled"]
    # c = arr[np.argmax(prediction)]

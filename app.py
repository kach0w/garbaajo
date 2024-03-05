import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
st.title("Garbage Classification App")
@st.cache(allow_output_mutation=True)
def load_model():
    model = Model()
    model.load_state_dict(torch.load("model.pth"))
    return model
def getPrediction(image):
    model = load_model()

    frame = Image.open(image)
    if frame.mode != "RGB":
        frame = frame.convert("RGB")
    frame = frame.resize((100, 100))
    frame = np.array(frame)
    st.image(frame)
    frame = np.transpose(frame, (2, 0, 1))
    frame = torch.tensor(frame, dtype = torch.float32).unsqueeze(0)

    outputs = model(frame)
    _, predictions = torch.max(outputs.data, 1)
    labels = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
    return labels[predictions.item()]

# frame = st.camera_input("Take Photo");
frame = st.file_uploader("Upload Photo")
if frame is not None:
    pred = getPrediction(frame)
    st.write(pred)

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import numpy as np
from PIL import Image, ImageOps

# Define the model architecture
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
model = DigitClassifier()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Streamlit UI setup
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("📤 Upload a Handwritten Digit Image")
st.write("Upload a **28x28 pixel grayscale** image of a digit (0–9).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", width=150)

    # Preprocess the image
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_np).view(-1, 28 * 28)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()

    # Show results
    st.subheader(f"🎯 Predicted Digit: {prediction}")
    st.write(f"Confidence: {confidence * 100:.2f}%")

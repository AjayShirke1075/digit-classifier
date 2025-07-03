import torch
from model import CNN
from utils import load_image

model = CNN()
model.load_state_dict(torch.load("saved_models/model.pth"))
model.eval()

image_path = "images/my_digit.png"
image_tensor = load_image(image_path)

with torch.no_grad():
    output = model(image_tensor)
    predicted = torch.argmax(output, 1)
    print(f"Predicted digit: {predicted.item()}")

import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import os
from torchvision import models
import torch.nn.functional as F

# Define the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Re-create the ImageClassifier model from your provided code
class ImageClassifier:
    def __init__(self):
        self.model = models.mobilenet_v3_small(pretrained=True)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=2)
        self.model.classifier.add_module('4', torch.nn.LogSoftmax(dim=1))
        self.model = self.model.to(device)
        self.model.eval()

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))

# Create an instance of the ImageClassifier
image_classifier = ImageClassifier()
# Load the trained model parameters
image_classifier.load('C:/Users/Axl Wynants/Desktop/Food-Image-Classifier/last-trained.pth')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Directory containing test images
test_images_dir = 'C:/Users/Axl Wynants/Desktop/Food-Image-Classifier/test_vgg/cropped_images'
test_image_paths = [os.path.join(test_images_dir, fname) for fname in os.listdir(test_images_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]

# Perform inference and measure time
def classify_image(model, image_tensor):
    with torch.no_grad():
        start_time = time.time()
        output = model(image_tensor)
        end_time = time.time()
        inference_time = end_time - start_time
        _, predicted_class = torch.max(output, 1)
    return predicted_class.item(), inference_time

# Store inference times
inference_times = []

# Process each test image
for image_path in test_image_paths:
    image_tensor = load_image(image_path)
    predicted_class, inference_time = classify_image(image_classifier.model, image_tensor)
    inference_times.append(inference_time)
    print(f"Image: {image_path}, Predicted Class: {predicted_class}, Inference Time: {inference_time:.6f} seconds")

# Calculate and print average inference time
average_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time: {average_inference_time:.6f} seconds")

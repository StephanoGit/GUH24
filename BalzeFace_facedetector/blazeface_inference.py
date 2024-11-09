import time

import nparray
import numpy as np
import torch
import cv2
from torchvision.models import VGG19_Weights
import torch.nn.functional as F
from blazeface import BlazeFace
import torch.nn as nn
from PIL import Image
import ssl
from torchvision import models, transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
ssl._create_default_https_context = ssl._create_unverified_context



vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT)
vgg19_encoder = nn.Sequential(*list(vgg19.features.children()))
vgg19_encoder.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(img: nparray) -> torch.Tensor:
    img = Image.fromarray(img).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Function to flatten the feature map into a vector
def flatten_features(features: torch.Tensor) -> torch.Tensor:
    """Flatten the feature map to a 1D vector."""
    return features.view(features.size(0), -1)  # Flatten all dimensions except batch

def extract_features(img_tensor):
    with torch.no_grad():  # No need for gradients for inference
        features = vgg19_encoder(img_tensor)
    return features


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

front_net = BlazeFace().to(gpu)
front_net.load_weights("blazeface.pth")
front_net.load_anchors("anchors.npy")
# front_net.min_score_thresh = 0.75
# front_net.min_suppression_threshold = 0.3




def detect_face(img):
    resized_image = cv2.resize(img, (128, 128))
    detections = front_net.predict_on_image(resized_image)
    faces = []
    for i in range(detections.shape[0]):
        ymin = int(detections[i, 0] * img.shape[0])
        xmin = int(detections[i, 1] * img.shape[1])
        ymax = int(detections[i, 2] * img.shape[0])
        xmax = int(detections[i, 3] * img.shape[1])

        faces.append([xmin, ymin, xmax, ymax])

    faces = np.array(faces)
    return faces



img_1 = cv2.imread("/Users/banika/Downloads/IMG_3149.png")
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
faces_1 = detect_face(img_1)
print(faces_1)

face1_subimg = img_1[faces_1[0][1]:faces_1[0][3], faces_1[0][0]:faces_1[0][2]]
face_1_features = load_image(face1_subimg)
face_1_features = extract_features(face_1_features)
face_1_features = flatten_features(face_1_features)

start_time = time.time()


img_2 = cv2.imread("/Users/banika/Downloads/IMG_3154.png")
faces_2 = detect_face(img_2)
print(faces_2)
face_2_subimg = img_2[faces_2[0][1]:faces_2[0][3], faces_2[0][0]:faces_2[0][2]]
face_2_features = load_image(face_2_subimg)
face_2_features = extract_features(face_2_features)
face_2_features = flatten_features(face_2_features)


# distance = F.pairwise_distance(face_2_features, face_1_features)

cosine_similarity = np.linalg.norm(face_2_features.numpy() - face_1_features.numpy())

similarity_percentage = ((cosine_similarity.item() + 1) / 2) * 100

stop_time = time.time()

print(f"Time taken: {stop_time - start_time:.4f} seconds")

print(f"Cosine Similarity: {cosine_similarity.item():.4f}")
print(f"Similarity Percentage: {similarity_percentage:.2f}%")




from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

# Load the FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Load two input images
image1_path = './detected_faces/face12.jpg'
image2_path = './detected_faces/face9.jpg'
image1 = Image.open(image1_path).convert('RGB')
image2 = Image.open(image2_path).convert('RGB')

# Detect faces and get embeddings for the images
mtcnn = MTCNN()
faces1 = mtcnn(image1)
faces2 = mtcnn(image2)
if faces1 is None or faces2 is None:
    print("No face detected in one or both images.")
    exit(1)

# Get embeddings for the faces
image1_embedding = facenet_model(faces1.unsqueeze(0)).detach().numpy()[0]
image2_embedding = facenet_model(faces2.unsqueeze(0)).detach().numpy()[0]

# Calculate cosine similarity between the embeddings
cosine_similarity = cosine(image1_embedding, image2_embedding)
print(f"Likliness of different person: {cosine_similarity * 100}%.")

# Set threshold for similarity, higher is less strict
similarity_threshold = 0.4

# Compare the similarity
if cosine_similarity < similarity_threshold:
    print("The two images depict the same person.")
else:
    print("The two images depict different persons.")

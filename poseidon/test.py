import cv2
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

video_capture = cv2.VideoCapture(0)
face_detector = MTCNN()

min_face_confidence = 0

# Load the detected faces from the directory
detected_faces = []
for i in range(1, 11):
    face_path = f"./poseidon/detected_faces/face{i}.jpg"
    face_image = Image.open(face_path)
    detected_faces.append((face_path, face_image))

while True:
    ret, frame = video_capture.read()
    pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    faces = face_detector.detect_faces(pixels)
    
    for face in faces:
        print(face['confidence'])
        if face['confidence'] > min_face_confidence:
            x, y, w, h = face['box']
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face embedding
            face_image = Image.fromarray(pixels[y:y+h, x:x+w])
            face_embedding = facenet_model(face_image).detach().numpy().flatten()
            
            # Compare with detected faces
            for detected_face in detected_faces:
                detected_face_path, detected_face_image = detected_face
                detected_face_embedding = facenet_model(detected_face_image).detach().numpy().flatten()
                similarity = cosine_similarity([face_embedding], [detected_face_embedding])[0][0]
                if similarity > 0.9:
                    print("Face detected!")
            
    cv2.imshow('frame', frame)
    
    # Stop the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
video_capture.release()
cv2.destroyAllWindows()

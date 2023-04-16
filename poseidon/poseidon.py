import cv2
from PIL import Image
from mtcnn.mtcnn import MTCNN
import numpy as np

video_capture = cv2.VideoCapture(0)
detector = MTCNN()

min_face_confidence = 0

while True:
    ret, frame = video_capture.read()
    pixels = np.asarray(frame)
    
    faces = detector.detect_faces(pixels)
    
    images_saved = 0

    for face in faces:
        print(face['confidence'])
        if face['confidence'] > min_face_confidence:
            x, y, w, h = face['box']
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # save the face to an image
            #TODO: change this to sqlite3
            new_image = Image.fromarray(pixels[y:y+h, x:x+h])
            images_saved += 1
            new_image.save(f"./poseidon/detected_faces/face{images_saved}.jpg")
            
            # facenet implementation
            

            if images_saved > 5:
                break
            
    cv2.imshow('frame', frame)
    
    # Stop the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the video capture device and close all windows
video_capture.release()
cv2.destroyAllWindows()
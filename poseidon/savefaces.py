import cv2
from PIL import Image
from mtcnn.mtcnn import MTCNN
import numpy as np

video_capture = cv2.VideoCapture('./poseidon/faceexamplevideo.mkv')
detector = MTCNN()

min_face_confidence = 0
images_saved = 0

delay_counter = 0

while True:
    ret, frame = video_capture.read()    
    
    if delay_counter == 60:
        delay_counter = 0
        
        faces = detector.detect_faces(frame)
        
        pixels = np.asarray(frame)
        faces = detector.detect_faces(pixels)

        for face in faces:
            print(face['confidence'])
            if face['confidence'] > min_face_confidence:
                x, y, w, h = face['box']
                
                imageRGB = cv2.cvtColor(pixels[y:y+h, x:x+h], cv2.COLOR_BGR2RGB)
                new_image = Image.fromarray(imageRGB)
                images_saved += 1
                new_image.save(f"./poseidon/detected_faces/face{images_saved}.jpg")
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if images_saved > 5:
                    print('faces limit reached')
                    break
    
            
    cv2.imshow('frame', frame)
    delay_counter += 1
    print(delay_counter)
    
    if images_saved > 20:
        break
    
    # Stop the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the video capture device and close all windows
video_capture.release()
cv2.destroyAllWindows()
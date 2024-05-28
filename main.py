# TODO:
#  - Send to an api when a match is found
#  - Save unmatched faces to a database as image or embedding
#  - maybe switch over to the seperate mtcnn
#  - when db grows, comparing faces will take more time than detecting faces
#    collect faces and only compare them set amount of sec/frames (batch processing)
#  - add compatibility for links to video not just a video file
#  - add embeddings to images?
#  - seperate modules
#  - fix importing modules multiple times
#  - seperate get_embeddings
#  - if tensor image doesnt have a face raise exception

# run detection, if match use naming scheme face{num}_{image number}

# TODO: Problems
# - rectange is not the right size
#  - not printing if an image doesnt contain a face

from common_imports import mtcnn

import cv2

from load_embeddings import load_embeddings
from face_matching import face_matching
from get_embedding import get_embedding
from save_face import save_face
from cl_args import cl_args

def main():
    images_to_load, images_path, video_file, min_probability, max_distance = cl_args()
    embeddings = load_embeddings(images_to_load, images_path)
    
    print('Finished loading faces')
    
    video_capture = cv2.VideoCapture(video_file)
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        frame_count += 1
        if not ret:
            print(f'Error loading video or video file ended (Frame: {frame_count})')
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        boxes, probs = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < min_probability:
                    continue
                x, y, w, h = box.astype(int)
                face = rgb_frame[y:y+h, x:x+w]
                face_embedding = get_embedding(face)
                match_info = face_matching(face_embedding, embeddings, max_distance)
                
                if match_info:
                    cosine_similarity, embedding_index, match_file = match_info
                    print(f'A face matched with {cosine_similarity * 100}% distance from embedding {embedding_index} in list (File: {match_file})')
                    print(f'Match found at frame {frame_count}')
                    cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)
                else:
                    # TODO: save as image or embedding or array or something
                    # need to fix match_file
                    match_file = False
                    cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (0, 0, 255), 2)
                      
                # convert face to PIL image maybe add this to save_face module
                face_file = save_face(face, match_file)
                print(f'Face saved to {face_file}')
        
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
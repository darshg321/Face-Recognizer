# TODO:
#  - Load face embeddings from images - X
#  - Use OpenCV and MTCNN to detect faces in video - X
#  - Use cosine distance to compare faces - X
#  - Send to an api when a match is found
#  - Save unmatched faces to a database as image or embedding
#  - maybe switch over to the seperate mtcnn
#  - use os to load images from a folder - X
#  - when db grows, comparing faces will take more time than detecting faces
#    collect faces and only compare them set amount of sec/frames (batch processing)
#  - maybe add argparse - X
#  - add compatibility for links to video not just a video file
#  - add support for gpu - X
#  - add embeddings to images ?
#  - seperate modules
#  - fix importing modules multiple times
#  - seperate get_embeddings
#  - if tensor image doesnt have a face raise exception


# run detection, if match use naming scheme face{num}_{image number}

# TODO: Problems
# - rectange is not the right size
#  - not printing if an image doesnt contain a face

from common_imports import os, mtcnn

import cv2
import argparse

from load_embeddings import load_embeddings
from face_matching import face_matching
from get_embedding import get_embedding
from save_face import save_face

def Main():

    parser = argparse.ArgumentParser(description='Find faces in a video and compare them to images of faces.\nSupported image formats: .jpg, .jpeg, .png.')
    parser.add_argument('--images_path', type=str, help='The path to the images folder')
    parser.add_argument('--load_amount', type=int, help='The amount of images to load after filtering out unsupported formats. Default is 10.')
    parser.add_argument('--video_path', type=str, help='The video file. Supported formats: .mkv, .mp4, .avi, .mov, .wmv.')
    parser.add_argument('--min_probability', type=float, help='The minimum probability for a face to be detected. Default is 0.95.\
    \nMax is 1. Higher values mean more strict detection.')
    parser.add_argument('--max_distance', type=float, help='The threshold for the distance of faces when comparing faces.\
    \nHigher values mean looser comparing, with a max of 1. Default is 0.4.')

    args = parser.parse_args()
    
    images_to_load = args.load_amount
    images_path = args.images_path
    video_file = args.video_path
    min_probability = args.min_probability
    max_distance = args.max_distance
    
    allowed_video_extensions = ['.mkv', '.mp4', '.avi', '.mov', '.wmv']
    
    # if all args are not specified, use example values
    if all(not arg for arg in [args.images_path, args.load_amount, args.video_path, args.min_probability, args.max_distance]):
        # maybe change these
        images_to_load = 3
        images_path = './detected_faces/'
        video_file = './faceexamplevideo.mkv'
        min_probability = 0.95
        max_distance = 0.4
        print('Args not specified, using example values')
    else:
        # if an arg is specified, but not all, print error and exit
        if not all(arg for arg in [args.images_path, args.load_amount, args.video_path]):
            parser.error('Missing required arguments')
    
    if not args.min_probability:
        min_probability = 0.95
        
    if not args.max_distance:
        max_distance = 0.4
    
    if not os.path.exists(video_file):
        raise Exception('Video file does not exist')
    if not any(video_file.endswith(ext) for ext in allowed_video_extensions):
        raise Exception('Video file extension not supported')
    
    
    # try:
    embeddings = load_embeddings(images_to_load, images_path)
    # except Exception as e:
        # print(e)
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
    Main()
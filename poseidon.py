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
#  - maybe add argparse

# TODO: Problems
# - rectange is not the right size

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import cv2
import os

facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN()

def load_embeddings(load_amount: int, images_path: str) -> dict:
    """
        Loads the face embeddings from the images in the images_path folder
        and returns a dict of the MTCNN embeddings, key being a tuple of the embeddings, 
        and value being the path of the image that corresponds.
        If there's no face detected in the image, it will raise an exception.
        Supported formats: .jpg, .jpeg, .png
        
        :param load_amount: The amount of images to load after filtering out unsupported formats
        :param images_path: The path to the images folder
        :return: A dict of the MTCNN embeddings, key is a tuple of the embedding, value is the image path
        :raises Exception: If there's no face detected in the image
        :raises Exception: If there's less images than the load amount
        :raises Exception: If the images folder doesn't exist
        
        :Example:
        
        >>> load_amount = 10
        >>> images_path = 'images/'
        >>> embeddings = load_embeddings(load_amount, images_path)
        
        >>> len(embeddings)
        10
        
        >>> print(embeddings)
        {(0, 0, 0, 0): 'image.jpg'}
    """
    
    if images_path[-1] != '/':
        images_path += '/'
    
    if not os.path.exists(images_path):
        raise Exception('Images path does not exist')
    
    allowed_extensions = ['jpg', 'jpeg', 'png']
    
    # filter the files, removing folders and extensions that aren't allowed
    filtered_files = [
        file for file in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, file)) and
        (any(file.endswith(ext) for ext in allowed_extensions) or
        print(f"File format not supported: {file}"))
    ]
    
    # if there are less files than the load amount, raise an exception
    if len(filtered_files) < load_amount:
        raise Exception(f'Supported files in the images folder lower than load amount, supported amount: {len(filtered_files)}, load amount: {load_amount}')
    
    images_embeddings: dict = {}
    for i, file in enumerate(filtered_files, start = 1):
        if i > load_amount:
            break
        
        # load the image
        face = Image.open(images_path + file).convert('RGB')
        face = mtcnn(face)
        if face is None:
            print('Face not detected in image', file)
            continue
        
        # get the embedding
        face_embedding = facenet_model(face.unsqueeze(0)).detach().numpy()[0]
        face_embedding = tuple(face_embedding.tolist())
        images_embeddings[face_embedding] = file
        
        print(f'Loaded face {i}: file {images_path + file}')
    
    return images_embeddings

def face_matching(face_embedding, embeddings: list | dict, similarity_threshold) -> bool:
    """
        Matches the face_embedding with the embeddings in the embeddings
        and returns True if a match is found.
        
        :param face_embedding: The MTCNN embedding of the face to match
        :param embeddings: A list or dictionary of MTCNN embeddings of the faces to match with. In a dictionary, the key has to be the embedding.
        :param similarity_threshold: The threshold to match the face with
        :return: True if a match is found, False otherwise.    
    """
    for i, embedding in enumerate(embeddings):
        cosine_similarity = cosine(face_embedding, embedding)
        
        if cosine_similarity < similarity_threshold:
            print(f'A face matched with {cosine_similarity * 100}% distance from embedding {i + 1} in list (File: {embeddings[embedding]})')
            return True
    
    return False


images_to_load = 15
images_path = './detected_faces/'

embeddings = load_embeddings(images_to_load, images_path)
print('Finished loading faces')

video_file = './faceexamplevideo.mkv'

video_capture = cv2.VideoCapture(video_file)
frame_count = 0
min_probability = 0.95

while True:
    ret, frame = video_capture.read()
    if not ret:
        print(f'Error loading video or video file ended (Frame: {frame_count})')
        break
    
    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    boxes, probs = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob < min_probability:
                continue
            x, y, w, h = box.astype(int)
            
            face = rgb_frame[y:y+h, x:x+w]
            # maybe needed
            face = cv2.resize(face, (160, 160))
            tensor_image = mtcnn(face)
            
            face_embedding = facenet_model(tensor_image.unsqueeze(0)).detach().numpy()[0]
            
            match = face_matching(face_embedding, embeddings, 0.4)
            if match:
                print(f'Match found at frame {frame_count}')
                # maybe save it
                cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)
            else:
                # TODO: save as image or embedding or array or something
                cv2.rectangle(frame, (x, y), ((x+w), (y+h)), (0, 0, 255), 2)
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
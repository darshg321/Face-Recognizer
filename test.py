# TODO:
#  - Load face embeddings from images - X
#  - Use OpenCV and MTCNN to detect faces in video - X
#  - Use cosine distance to compare faces - X
#  - Send to an api when a match is found
#  - Save unmatched faces to a database as image or embedding

# TODO: Problems
# - rectange is not the right size
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
import cv2

facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN()


def load_embeddings(load_amount: int, images_path: str) -> list:
    """
    Loads the face embeddings from the images in the images_path folder
    and returns a list of the MTCNN embeddings.
    If there's no face detected in the image, it will raise an exception.
    
    :param load_amount: The amount of images to load
    :param images_path: The path to the images folder
    :return: A list of the MTCNN embeddings of the loaded faces.
    :exception: If there's no face detected in the image, it will raise an exception.
    
    Example:
    
    load_amount = 5
    
    images_path = './detected_faces/'
    
    embeddings = load_embeddings(load_amount, images_path)
    
    print(embeddings)
    """
    embeddings = []
    for i in range(1, load_amount + 1):
        image = images_path + 'face' + str(i) + '.jpg'
        
        face = Image.open(image).convert('RGB')
        face = mtcnn(face)
        if face is None:
            print('Face not detected in image', image)
            raise Exception('Face not detected in image')
        
        face = facenet_model(face.unsqueeze(0)).detach().numpy()[0]
        embeddings.append(face)
        print('Loaded face', i)
    
    return embeddings

images_to_load = 15
images_path = './detected_faces/'

embeddings = load_embeddings(images_to_load, images_path)
print('Finished loading faces')

def face_matching(face_embedding, embedding_list: list, similarity_threshold) -> bool:
    """
    Matches the face_embedding with the embeddings in the embedding_list
    and returns True if a match is found.
    
    :param face_embedding: The MTCNN embedding of the face to match
    :param embedding_list: The list of MTCNN embeddings of the faces to match with
    :param similarity_threshold: The threshold to match the face with
    :return: True if a match is found, False otherwise.
    
    Example:
    
    face_embedding = embeddings[0]
    
    embedding_list = embeddings[1:]
    
    similarity_threshold = 0.4
    
    face_matching(face_embedding, embedding_list, similarity_threshold)
    """
    
    # maybe return what face its matching with specifically or print similarity with all faces
    # maybe return a list with distances and check each distance, instead of bool
    for i, embedding in enumerate(embedding_list):
        cosine_similarity = cosine(face_embedding, embedding)
        if cosine_similarity < similarity_threshold:
            # this uses precoded file name
            print(f'A face matched with {cosine_similarity * 100}% distance from embedding {i + 1} in list (file face{i + 1}.jpg)')
            return True
    
    return False

video_file = './faceexamplevideo.mkv'

video_capture = cv2.VideoCapture(video_file)
min_probability = 0.95

while True:
    ret, frame = video_capture.read()
    if not ret:
        print('Error loading video or video file ended')
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    boxes, probs = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob < min_probability:
                continue
            x, y, w, h = box
            x, y, w, h = round(x), round(y), round(w), round(h)

            face = rgb_frame[y:y+h, x:x+w]
            # maybe needed
            # face = cv2.resize(face, (160, 160))
            tensor_image = mtcnn(face)
            
            # TODO: save face as image or maybe face embedding
            
            face_embedding = facenet_model(tensor_image.unsqueeze(0)).detach().numpy()[0]
            
            match = face_matching(face_embedding, embeddings, 0.4)
            if match:
                print('Match found')
                cv2.rectangle(frame, (x, y), (round((x+w)/1.5), round((y+h)/1.5)), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (round((x+w)/1.5), round((y+h)/1.5)), (0, 255, 0), 2)
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
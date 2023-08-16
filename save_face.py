from random import randint
from common_imports import Image

def save_face(face_image, match_file):
    # change from random to some system
    # face_image needs to be array or pil image
    # maybe resize ?
    
    face_image = Image.fromarray(face_image)
    
    if not match_file:
        match_file = 'unknown'
    filename = f'./saved_faces/{match_file}_{randint(1, 1000)}.jpg'
    face_image.save(filename)
    
    return filename
from common_imports import Image, os, facenet_model, mtcnn
from get_embedding import get_embedding

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
    
    allowed_image_extensions = ['jpg', 'jpeg', 'png']
    
    # filter the files, removing folders and extensions that aren't allowed
    filtered_files = [
        file for file in os.listdir(images_path)
        if os.path.isfile(os.path.join(images_path, file)) and
        (any(file.endswith(ext) for ext in allowed_image_extensions) or
        print(f"File format not supported: {file}"))
    ]
    
    # if there are less files than the load amount, raise an exception
    # if len(filtered_files) < load_amount:
    #     raise Exception(f'Supported files in the images folder lower than load amount, supported amount: {len(filtered_files)}, load amount: {load_amount}')
    
    images_embeddings: dict = {}
    for i, file in enumerate(filtered_files, start = 1):
        if i > load_amount:
            break
        
        # load the image
        face = Image.open(images_path + file).convert('RGB')
        
        face_embedding = get_embedding(face)

        if face_embedding is None:
            raise Exception(f'Face not detected in image {file}')
        
        # face = mtcnn(face)
        # if face is None:
        #     print('Face not detected in image', file)
        #     continue
        
        # # get the embedding
        # face_embedding = facenet_model(face.unsqueeze(0)).detach().numpy()[0]
        face_embedding = tuple(face_embedding.tolist())
        images_embeddings[face_embedding] = file
        
        print(f'Loaded face {i}: file {images_path + file}')
    
    return images_embeddings
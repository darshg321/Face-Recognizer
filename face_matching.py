from scipy.spatial.distance import cosine

def face_matching(face_embedding, embeddings: list | dict, similarity_threshold: float):
    """
        Matches the face_embedding with the embeddings in the embeddings
        and returns True if a match is found.
        
        :param face_embedding: The MTCNN embedding of the face to match
        :param embeddings: A list or dictionary of MTCNN embeddings of the faces to match with.
        In a dictionary, the key has to be the embedding.
        :param similarity_threshold: The threshold to match the face with
        :return: A tuple containing the cosine similarity in percent, the index of the embedding in the list and the
        name of the file if a match is found, False otherwise.
    """
    for i, embedding in enumerate(embeddings):
        cosine_similarity = cosine(face_embedding, embedding)
        
        if cosine_similarity < similarity_threshold:
            info = (cosine_similarity * 100, i + 1, embeddings[embedding])
            return info
    
    return False
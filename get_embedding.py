from common_imports import mtcnn, facenet_model

def get_embedding(face):
    # face is an rgb frame
    tensor_image = mtcnn(face)
    
    if tensor_image is None:
        return None
                
    return facenet_model(tensor_image.unsqueeze(0)).detach().numpy()[0]
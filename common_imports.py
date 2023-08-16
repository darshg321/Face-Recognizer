from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from torch.cuda import is_available

facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
device = 'cuda' if is_available() else 'cpu'
mtcnn = MTCNN(device=device)
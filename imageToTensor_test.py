import torch
from torch.autograd import Variable
from imageToTensor import image_to_tensor
from loadPictures import style_path, content_path

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'cpu')

def preprocess (path):
    img = image_to_tensor(path)
    img = Variable(img).to(device)
    return img

def test():
    style_img =preprocess(style_path)
    content_img = preprocess(content_path)
    print(style_img.size(), content_img.size())


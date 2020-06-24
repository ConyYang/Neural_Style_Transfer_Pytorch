import torch
from torch.autograd import Variable
from imageToTensor import image_to_tensor
from loadPictures import style_path, content_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

style_img = image_to_tensor(style_path)
style_img = Variable(style_img).to(device)

content_img = image_to_tensor(content_path)
content_img = Variable(content_img).to(device)

print(style_img.size(), content_img.size())
from LossFunction.contentLoss import Content_Loss
from imgProcess.imageToTensor_test import preprocess, device
from imgProcess.loadPictures import content_path
import torch

def test():
    content_img = preprocess(content_path)
    cl = Content_Loss(content_img, 1)

    rand_img = torch.randn(size=content_img.data.size(),
                           device=device)
    cl.forward(rand_img)
    print(cl.loss)
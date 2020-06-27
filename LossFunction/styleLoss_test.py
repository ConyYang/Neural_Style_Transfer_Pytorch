from imgProcess.loadPictures import style_path
from LossFunction.styleLoss import Gram, Style_Loss
from imgProcess.imageToTensor_test import preprocess, device
import torch
gram = Gram()

def test():
    style_img = preprocess(path=style_path)
    graph = gram(style_img)

    sl = Style_Loss(graph, 1000)
    rand_img = torch.randn(size = style_img.size(), device=device)

    sl.forward(rand_img)
    print(sl.loss)

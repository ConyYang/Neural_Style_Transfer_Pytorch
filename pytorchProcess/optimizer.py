import torch.optim as optim
import torch.nn as nn


def get_input_param_optimier(input_img):
    # transfer input image to a parameter type
    input_param = nn.Parameter(input_img.data)
    # optimizer input_img, not weight
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

from LossFunction.styleLoss_test import Gram
from LossFunction.contentLoss import Content_Loss
from LossFunction.styleLoss import Style_Loss
import torch.nn as nn
from imgProcess.imageToTensor_test import device

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

model = nn.Sequential()
model = model.to(device)


def get_style_model_and_loss(style_img, content_img, cnn,
                             style_weight=1000, content_weigt=1,
                             content_layers = content_layers_default,
                             style_layers = style_layers_default):
    # store the 6 loss functions
    content_loss_list = []
    style_loss_list = []

    gram = Gram()
    gram = gram.to(device)

    i =1
    for layer in cnn:
        if isinstance(layer, nn.Conv2d):
            name = 'conv_'+str(i)
            model.add_module(name, layer) # add

            # check if conv layer should use to calculate the loss
            if name in content_layers:
                # put target into model, get target
                target = model(content_img)
                content_loss = Content_Loss(target, content_weigt)
                model.add_module('content_loss_'+str(i), content_loss) #add
                content_loss_list.append(content_loss)

            # extract style
            if name in style_layers:
                target = model(style_img)
                target = gram(target)
                style_loss = Style_Loss(target, style_weight)
                model.add_module('style_loss_'+str(i), style_loss) # add
                style_loss_list.append(style_loss)

            i +=1

        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_'+str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)

    return model, content_loss_list, style_loss_list

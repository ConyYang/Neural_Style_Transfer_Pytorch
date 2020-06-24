# VGG19 Neural Network
import torchvision.models as models
from imageToTensor_test import device
import torch.nn as nn
import torch
from styleLoss_test import Gram
from contentLoss import Content_Loss
from styleLoss import Style_Loss
from loadPictures import content_path, style_path
from imageToTensor_test import preprocess

torch.utils.model_zoo.load_url("https://labfile.oss.aliyuncs.com/courses/861/vgg19_pre.zip")
vgg = models.vgg19(pretrained=True).features
vgg = vgg.to(device)

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

model = nn.sequential()
model = model.to(device)

def get_style_model_and_loss(style_img, content_img, cnn=vgg,
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
            if name in content_layers_default:
                # put target into model, get target
                target = model(content_img)
                content_loss = Content_Loss(target, content_weigt)
                model.add_module('content_loss_'+str(i), content_loss) #add
                content_loss_list.append(content_loss)

            # extract style
            if name in style_layers_default:
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


content_img = preprocess(content_path)
style_img = preprocess(style_path)

model, content_loss_list, style_loss_list = get_style_model_and_loss(style_img=style_img, content_img=content_img)

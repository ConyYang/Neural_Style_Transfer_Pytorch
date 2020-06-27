# VGG19 Neural Network
import torchvision.models as models
from imgProcess.imageToTensor_test import device
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from imgProcess.loadPictures import content_path, style_path
from imgProcess.imageToTensor_test import preprocess
from pytorchProcess.model import get_style_model_and_loss
from pytorchProcess.train import run_style_transfer

vgg = models.vgg19(pretrained=True).features
vgg = vgg.to(device)

content_img = preprocess(content_path)
style_img = preprocess(style_path)

model, content_loss_list, style_loss_list = get_style_model_and_loss(style_img=style_img, content_img=content_img,
                                                                     cnn=vgg)
# initialize G
input_img = content_img.clone()
# train model and return img
out = run_style_transfer(content_img, style_img, input_img, num_epochs=3, model=model,
                         content_loss_list=content_loss_list, style_loss_list=style_loss_list)
new_pic = transforms.ToPILImage()(out.cpu().squeeze(0))
print("finish")

fig, ax = plt.subplots()
im = ax.imshow(new_pic)
plt.show()
plt.savefig("new_pic.jpg")
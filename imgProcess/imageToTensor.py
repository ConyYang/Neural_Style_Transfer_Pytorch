import PIL.Image as Image
import torchvision.transforms as transforms
img_size = 512


def image_to_tensor(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img


from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import numpy as np
from PIL import Image
import torch

def img2vec(image,image_size = 128):
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(), # turn into Numpy array of shape HWC, divide by 255
        Lambda(lambda t: (t * 2) - 1),
    ])
    return transform(image).unsqueeze(0)
 

def vec2img(vec):
    vec = vec.squeeze(0)
    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])
    return reverse_transform(vec)

def pil2tensor(item):
    tensor_img = torch.from_numpy(np.array(item)).permute(2, 0, 1).float()/255.0
    return tensor_img

def tensor2pil(item):
    tensor = item
    tensor = tensor.cpu().clone()
    tensor = tensor.squeeze(0)
    tensor = tensor.permute(1, 2, 0)
    image = tensor.numpy()
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image

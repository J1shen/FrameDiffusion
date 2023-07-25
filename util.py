from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import numpy as np

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
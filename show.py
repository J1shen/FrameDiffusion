from PIL import Image
import requests
import torch
from dataprocess import img2vec,vec2img
from diffusion import DiffusionModel

 
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
image.save('test.jpg')

vec = img2vec(image)

model = DiffusionModel()
for i in [0, 50, 100, 150, 199]:
    print(i)
    img = model.get_noisy_image(vec,torch.tensor([i]))
    img.save(f'noise_{i}.jpg')
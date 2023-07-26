from PIL import Image
import requests
import torch,random
import numpy as np
from util import img2vec,vec2img
from ultralytics import YOLO
from diffusion import DiffusionModel

'''
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
image.save('test.jpg')

vec = img2vec(image)

model = DiffusionModel()
for i in [0, 50, 100, 150, 199]:
    img = model.get_noisy_image(vec,torch.tensor([i]))
    img.save(f'noise_{i}.jpg')

'''
yolomodel = YOLO("yolov8n-pose.pt")
person = Image.open('person.jpg')
results = yolomodel(person)
#print(results[0].boxes.xyxy.numpy())
print(results[0].keypoints.xy[0])
res_plotted = Image.fromarray(results[0].plot())
res_plotted.save('detect.jpg')
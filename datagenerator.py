from datasets import load_dataset
from ultralytics import YOLO
from PIL import Image
import random

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

def crop_image(image, center, crop_size):
    left = max(0, center[0] - crop_size[0] // 2)
    right = min(image.size[0], center[0] + crop_size[0] // 2)
    top = max(0, center[1] - crop_size[1] // 2)
    bottom = min(image.size[1], center[1] + crop_size[1] // 2)
    return image.crop((left, top, right, bottom))
  
def generate_data(example):
  img_size = (256,256)
  image = Image.fromarray(example['image'])
  example['image'] = np.asarray(image.resize(img_size, resample=Image.BICUBIC))
  result = model(image)
  example['ori_keys'] = result.keypoints
  
  ratio_w = random.uniform(0.9, 1)
  ratio_h = random.uniform(1, 1.2)
  new_size = (int(256*ratio_w), int(256*ratio_h))
  new_img = image.resize(new_size, resample=Image.BICUBIC)

  boxes = result.boxes
  center = (boxes[0][0] + boxes[0][2]) // 2, (boxes[0][1] + boxes[0][3]) // 2  # extract center point from box
  
  corped = crop_image(new_img, center, 256).resize(img_size, resample=Image.BICUBIC)
  example['image_trans'] = np.asarray(croped)
  example['key_trans'] = model(corped).keypoints
  return example
  
  
dataset = load_dataset("fuliucansheng/pascal_voc")
new_dataset = dataset.filter(lambda example: 14 in example['classes'])

image_dataset = new_dataset.map(lambda example: example['image'])


keypoints = result.keypoints

if __name__ = '__main__':
  pass


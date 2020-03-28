import os

import pandas as pd
import numpy as np
from skimage import io
from skimage.transform import resize

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

data_transform = transforms.Compose([
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

IMG_WIDTH = 448
IMG_HEIGHT = 448
CELL = 7
CLASS = 20
BBOX = 2
IMAGE_PATH = 'C:/Users/adelk/PycharmProjects/yolo_v1/VOC2012/JPEGImages'
device = 'cuda'

# Парсим лейблы
def labels_parse(file, idx):
    data = file.iloc[idx, :]
    labels = []
    temp_w = 1.0 / int(data['width'])
    temp_h = 1.0 / int(data['height'])
    for count in range(int(data['obj'])):
        label = []
        label.append(((float(data[f'bbox_{count}_xmin']) + float(data[f'bbox_{count}_xmax'])) / 2.0) * temp_w) # центр
        label.append(((float(data[f'bbox_{count}_ymin']) + float(data[f'bbox_{count}_ymax'])) / 2.0) * temp_h)
        label.append((float(data[f'bbox_{count}_xmax']) - float(data[f'bbox_{count}_xmin'])) * temp_w) # ширина и высота бокса
        label.append((float(data[f'bbox_{count}_ymax']) - float(data[f'bbox_{count}_ymin'])) * temp_h)
        label.append(classes.index(data[f'bbox_{count}_name']))
        labels.append(label)
    return labels


# Конвертируем лейблы в тензор
def lb_to_tensor(label_arr):
    gtTensor = torch.zeros(CELL, CELL, 5).to(device)
    for label in label_arr:
        x, y, w, h, c = label[0], label[1], label[2], label[3], label[4]
        gtTensor[int(y * CELL), int(x * CELL)] = torch.tensor([x, y, w, h, c], dtype=torch.float64).to(device)
    return gtTensor


class VOCPascal(Dataset):

    def __init__(self, csv_file, width, height, transform=None):
        self.main_file = csv_file
        self.transform = transform
        self.width = width
        self.height = height

    def __len__(self):
        return self.main_file.shape[0]

    def __getitem__(self, index):
        image_name = self.main_file.iloc[index]['filename']
        image = io.imread(os.path.join(IMAGE_PATH, image_name))
        image = resize(image, (self.width, self.height))
        image = self.transform(torch.Tensor(image).permute(2, 0, 1))
        label = lb_to_tensor(labels_parse(self.main_file, index))
        return image, label, image_name
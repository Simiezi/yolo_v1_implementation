import time
import os
import copy

import torch
from torch.utils.data import DataLoader
import torch.nn
import torch.optim as optim
import torchvision.transforms as transforms

from tqdm import tqdm_notebook
import pandas as pd

from dataset import VOCPascal
from model import YoloV1NN
from loss_yolo import *


IMG_WIDTH = 448
IMG_HEIGHT = 448
SIZE = 7
CLASS = 20
BBOX = 2
BATCH_SIZE = 2

if __name__ == '__main__':

  model = YoloV1NN()

  data_transform = transforms.Compose([
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
  ])

  torch.cuda.empty_cache()
  data = pd.read_csv('C:/Users/adelk/PycharmProjects/yolo_v1/VOC2012/annotations.csv')
  train_data = VOCPascal(data.iloc[:10, :], IMG_WIDTH, IMG_HEIGHT, data_transform)
  val_data = VOCPascal(data.iloc[5201:5901, :], IMG_WIDTH, IMG_HEIGHT, data_transform)
  dataloaders = dict()
  dataloaders['train'] = DataLoader(train_data, batch_size=BATCH_SIZE)
  dataloaders['val'] = DataLoader(val_data, batch_size=BATCH_SIZE)
  learning_rate = 1e-4
  device = torch.device("cpu")
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
  num_epochs=10
  phase = 'train'

#   Y_out = model(X)
#
#   total_loss = full_loss(Y_out, Y)
#   print(f'Total loss = {total_loss}')
#
#   for t in range(30):
#     Y_out = model(X)
#     loss = full_loss(Y_out.clone(), Y.clone())
#     print(f'\n Epoch = {t + 1} Loss = {loss.item()}')
#     optimizer.zero_grad()
#     loss.backward()+
  for epoch in range(num_epochs):
    model.train()
    since = time.time()
    print(f'Epoch {epoch + 1}/{num_epochs}')
    y_out_epoch = torch.Tensor().to(device)
    running_loss = 0.0
    image_names = []
    for i, (image_batch, label_batch, image_name) in enumerate(dataloaders['train']):
      image_batch = image_batch.to(device)
      label_batch = label_batch.to(device)
      # image_names.append(image_name) для валидации
      optimizer.zero_grad()

      with torch.set_grad_enabled(phase == 'train'):
        y_out = model(image_batch)
        y_out_epoch = torch.cat((y_out_epoch, y_out), 0)
        y_out_epoch = y_out_epoch.to(device)

        loss = full_loss(y_out.clone(), label_batch.clone())
        print(loss)
        running_loss += loss.item() * image_batch.size(0)

        scheduler.step(loss)

        loss.backward()
        optimizer.step()
        print(f'Epoch={epoch} \t Batch={i} \t Loss={loss.item():.4f}\n')
        print(time.time() - since)
#     optimizer.step()
# print('Done')
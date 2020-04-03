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
from loss import *
from dataset import VOCPascal
from model import YoloV1NN



IMG_WIDTH = 448
IMG_HEIGHT = 448
SIZE = 7
CLASS = 20
BBOX = 2
BATCH_SIZE = 32

if __name__ == '__main__':
  data_transform = transforms.Compose([
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
  ])

  model = YoloV1NN()
  torch.cuda.empty_cache()
  model.cuda()
  device = torch.device("cuda")
  data = pd.read_csv('C:/Users/adelk/PycharmProjects/yolo_v1/VOC2012/annotations.csv')
  shuffled_data = data.sample(frac=1)
  train_data = VOCPascal(shuffled_data.iloc[:500, :], IMG_WIDTH, IMG_HEIGHT, data_transform)
  val_data = VOCPascal(shuffled_data.iloc[5201:5901, :], IMG_WIDTH, IMG_HEIGHT, data_transform)
  dataloaders = dict()
  dataloaders['train'] = DataLoader(train_data, batch_size=BATCH_SIZE)
  dataloaders['val'] = DataLoader(val_data, batch_size=BATCH_SIZE)
  learning_rate = 1e-5
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
  num_epochs = 80
  phase = 'train'

  for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    y_out_epoch = torch.Tensor().to(device)
    running_loss = 0.0
    image_names = []
    model.train()
    since = time.time()
    for i, (image_batch, label_batch, image_name) in enumerate(dataloaders['train']):
      image_batch = image_batch.to(device)
      label_batch = label_batch.to(device)
      optimizer.zero_grad()

      with torch.set_grad_enabled(phase == 'train'):
        y_out = model(image_batch)
        loss = full_loss(y_out.clone(), label_batch.clone())
        running_loss += loss.item() * image_batch.size(0)
        # scheduler.step(loss)
        loss.backward()
        optimizer.step()
        print(f'Batch={i + 1}\t Loss={loss.item():.4f}')
    # tb.save_value('Train loss', 'train_loss', epoch, loss.item())
    if epoch + 1 % 5 == 0:
      torch.save(model.state_dict(), f'/content/drive/My Drive/checkpoint{epoch + 1}.pth')
    print('\n')
    print(f'Epoch={epoch + 1} Loss={running_loss / len(dataloaders[phase].dataset):.4f}\n')
    print(f'Time of epoch {epoch + 1}={time.time() - since}\n')
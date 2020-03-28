import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import pandas as pd
from skimage import io
from skimage.transform import resize
import torch


data = pd.read_csv('//VOCdevkit/VOC2012/annotations.csv')

path = '//VOCdevkit/VOC2012/JPEGImages'
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# mmemmm = data.iloc[2, : ]

def labels_parse(file, idx):
    data = file.iloc[idx, : ]
    labels = []
    temp_w = 1.0/int(data['width'])
    temp_h = 1.0/int(data['height'])
    for count in range(int(data['obj'])):
        label = []
        label.append(((float(data[f'bbox_{count}_xmin']) + float(data[f'bbox_{count}_xmax'])) / 2.0) * temp_w)
        label.append(((float(data[f'bbox_{count}_ymin']) + float(data[f'bbox_{count}_ymax'])) / 2.0) * temp_h)
        label.append((float(data[f'bbox_{count}_xmax']) - float(data[f'bbox_{count}_xmin'])) * temp_w)
        label.append((float(data[f'bbox_{count}_ymax']) - float(data[f'bbox_{count}_ymin'])) * temp_h)
        label.append(classes.index(data[f'bbox_{count}_name']))
        labels.append(label)
    return labels

# for i in range(10):
#     print(labels_parse(data, i))
#     print('\n')

# labl = labels_parse(data, 2)
# image = resize(io.imread(os.path.join(path, mmemmm['filename'])), 448, 448)
# io.imshow(image)
# image = torch.Tensor(image)
# print(image)
# img_path = os.path.join(path, mmemmm['filename'])
# image = io.imread(img_path)
# image = resize(image, (448, 448))
# io.imshow(image)
# io.show()
# image = torch.Tensor(image)
# print(image)
# print('\n')
# print(image.size())
# print('\n')
# image = image.permute(2, 0, 1)
# print(image)
# print('\n')
# print(image.size())
# labels = labels_parse(data, 5)
# labels = labels_parse(data, 5) + 0.05
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



### box1 - gt box, box2 - prediction box

def IOU(box1, box2):
    x_min1 = torch.clamp(box1[0] - box1[2] / 2, 0, 1).to(device)
    x_max1 = torch.clamp(box1[0] + box1[2] / 2, 0, 1).to(device)
    y_min1 = torch.clamp(box1[1] - box1[3] / 2, 0, 1).to(device)
    y_max1 = torch.clamp(box1[1] + box1[3] / 2, 0, 1).to(device)

    x_min2 = torch.clamp(box2[0] - box2[2] / 2, 0, 1).to(device)
    x_max2 = torch.clamp(box2[0] + box2[2] / 2, 0, 1).to(device)
    y_min2 = torch.clamp(box2[1] - box2[3] / 2, 0, 1).to(device)
    y_max2 = torch.clamp(box2[1] + box2[3] / 2, 0, 1).to(device)

    point_a = torch.min(x_max1, x_max2)
    point_b = torch.max(x_min1, x_min2)
    point_c = torch.min(y_max1, y_max2)
    point_d = torch.max(y_min1, y_min2)

    width = torch.max(point_a - point_b, 0)
    height = torch.max(point_c - point_d, 0)

# since = time.time()
  #
  # best_model_weights = copy.deepcopy(model.state_dict())
  # best_acc = 0.0
  #
  # for epoch in range(num_epochs):
  #   print('Epoch {}/{}'.format(epoch, num_epochs - 1))
  #   print('-' * 10)
  #
  #   # Тренируем и валидируем на каждой эпохе
  #   for phase in ['train', 'val']:
  #     if phase == 'train':
  #       scheduler.step()
  #       model.train()
  #     else:
  #       model.eval()
  #
  #     running_loss = 0.0
  #     running_corrects = 0
  #
  #     for inputs, labels in dataloaders[phase]: #dataloader - будет после написания датасета, в виде словаря
  #       inputs = inputs.to(device)
  #       labels = labels.to(device)
  #       optimizer.zero_grad()
  #       # обучаемся и записываем историю, если в режиме обучения
  #       with torch.set_grad_enabled(phase == 'train'):
  #         outputs = model(inputs)
  #         _, preds = torch.max(outputs, 1)
  #         loss = full_loss(outputs, labels)
  #         # Бэкпроп
  #         if phase == 'train':
  #           loss.backward()
  #           optimizer.step()
  #       running_loss += loss.item() * inputs.size(0)
  #       running_corrects += torch.sum(preds == labels.data)
  #
  #     epoch_loss = running_loss / dataset_sizes[phase] #Тоже будет после написания датасета
  #     epoch_acc = running_corrects.double() / dataset_sizes[phase]
  #
  #     print('{} Loss: {:.4f} Acc: {:.4f}'.format(
  #       phase, epoch_loss, epoch_acc))
  #
  #     # сохраняем весе в случае
  #     if phase == 'val' and epoch_acc > best_acc:
  #       best_acc = epoch_acc
  #       best_model_weights = copy.deepcopy(model.state_dict())
  #
  #   print()
  #
  # time_elapsed = time.time() - since
  # print('Обучение завершено за {:.0f}m {:.0f}s'.format(
  #   time_elapsed // 60, time_elapsed % 60))
  # print('Лучшая валидация: {:4f}'.format(best_acc))
  #
  #
  # model.load_state_dict(best_model_weights)


if __name__ == '__main__':
    mem = data.iloc[:5201, :]
    print(mem.iloc[10]['filename'])
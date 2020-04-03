import torch.nn as nn
import torch
import torch.nn.functional as F

IMG_WIDTH = 448
IMG_HEIGHT = 448
SIZE = 7
CLASS = 20
BBOX = 2


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class YoloV1NN(nn.Module):

    def __init__(self, weights=None):

        super(YoloV1NN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),


            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),


            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )

        self.flatten = Flatten()

        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=SIZE * SIZE * 1024, out_features=4096),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=SIZE * SIZE * (BBOX * 5 + CLASS))
        )

        if weights is None:
            self.weights_init()

    def weights_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, images):
        features = self.cnn_layers(images)
        flatten = self.flatten(features)
        result = self.fc_layer(flatten)
        result = result.view(result.size(0), SIZE, SIZE, BBOX * 5 + 20)
        return result

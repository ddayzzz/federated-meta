import torch
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, num_classes, image_size):
        super(Model, self).__init__()
        self.input_shape = (1, image_size * image_size)
        self.num_classes = num_classes
        self.image_size = image_size
        # 创建参数
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        # self.fc1 = nn.Linear(1024, 2048)
        self.fc1 = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = x.view(-1, 1, self.image_size, self.image_size)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x



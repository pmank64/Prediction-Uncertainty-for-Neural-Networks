import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF


class simpleNet(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layer1 = torch.nn.Sequential(
         torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
         torch.nn.BatchNorm2d(num_features = 16),
         torch.nn.ReLU()
         # torch.nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.fc_layer2 = torch.nn.Sequential(
        # torch.nn.Linear(sizeOutChannels2*750*1000, sizeHiddenLayer),
        torch.nn.Linear(16*375*500, 2)
        # torch.nn.LeakyReLU(),
        # decide if we need this later?
        # torch.nn.Dropout(p=0.2),
        # return a set of two values at the end representing the center of the lane
        # torch.nn.Linear(1000, 1)
    )  

  def forward(self, x):
    x = self.conv_layer1(x)
    x = x.view(x.size(0), -1)
    x = self.fc_layer2(x)
    return x
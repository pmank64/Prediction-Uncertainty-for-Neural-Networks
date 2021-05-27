import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

class roadNet(torch.nn.Module):
  def __init__(self, sizeOutChannels, sizeHiddenLayer, outChannels, x, y):
    self.init_args = {k:v for k,v in locals().items() if k!='self' and k!='__class__'}
    super(roadNet, self).__init__()
    y = int(y / 2)
    x = int(x / 2)
    self.conv_layer = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=3, out_channels=sizeOutChannels, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(num_features = sizeOutChannels),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.fc_layer = torch.nn.Sequential(
        torch.nn.Linear(sizeOutChannels*y*x, sizeHiddenLayer),
        torch.nn.ReLU(),
        # decide if we need this later?
        torch.nn.Dropout(p=0.2),
        # return a set of two values at the end representing the center of the lane
        torch.nn.Linear(sizeHiddenLayer, outChannels)
    )  

  def forward(self, x):
    # conv_layer
    x = self.conv_layer(x)
    print(x.shape)
    # flatten
    x = x.view(x.size(0), -1)
    # fc_layer
    x = self.fc_layer(x)
    return x
  
  def load(self,model,opt,path):
    torch.save({'model_class': model.__class__,
              'model_args': model.init_args, 
              'model_state_dict': model.state_dict(),
              'opt_class': opt.__class__,
              'opt_args': opt.defaults,
              'opt_state_dict':opt.state_dict()},
            path)

  def save(self,model,opt, path):
      torch.save({'model_class': model.__class__,
              'model_args': model.init_args, 
              'model_state_dict': model.state_dict(),
              'opt_class': opt.__class__,
              'opt_args': opt.defaults,
              'opt_state_dict':opt.state_dict()},
            path)
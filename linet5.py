import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

class LeNet5(torch.nn.Module):

    def __init__(self, out):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = torch.nn.Sequential(            
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            torch.nn.Tanh()
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=84, out_features=out),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        #probs = F.softmax(logits, dim=1)
        return x
import torch
import pandas as pd
from torchvision.io import read_image
import os
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform, img_data):
        self.img_labels = pd.read_csv(annotations_file, header = None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_data = img_data

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, 'road' + str(idx) + '.png')
        # image = read_image(img_path)

        # get the image data out of the image data list
        image = self.img_data[idx]

        # get truth value out of annotations file
        label = self.img_labels.iloc[idx, 1]

        temp2 = []
        if self.transform:
            image = self.transform(image)
            image = image.permute(1, 2, 0)
        if self.target_transform:
            m = 0
        temp = label.strip('][').split(', ')
        for item2 in temp:
            temp2.append(float(item2))
        return [image, torch.tensor(temp2)]




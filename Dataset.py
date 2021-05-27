import torch
import pandas as pd
from torchvision.io import read_image
import os
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 'road' + str(idx) + '.png')
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        #print(label)
        if self.transform:
            image = self.transform(image.numpy())
            image = torch.reshape(image, (3,750,1000))
        if self.target_transform:
            temp2 = []
            # temp = label.strip('][').split(', ')
            # for item2 in temp:
            #     temp2.append(float(item2))
            # label = self.target_transform(np.array(temp2, dtype=np.float32))
        temp2 = []
        temp = label.strip('][').split(', ')
        for item2 in temp:
            temp2.append(float(item2))
        sample = {"image": image, "label": label}
        #print(idx)
        return [image, label]




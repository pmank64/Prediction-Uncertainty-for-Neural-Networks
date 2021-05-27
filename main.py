from road_image import roadImage
import numpy as np

import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms

from roadNet import roadNet

from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt

from RoadImage import *

from loader import *

from datetime import datetime

from torch.utils.data.sampler import SubsetRandomSampler

from linet5 import *

# set initial values
horizon = 250
road_width = 500
shoulder = 20
offset = 40
rotation = 0
paint_width = 5
img_width = 1000
img_height = 750
path = 'C:/Users/pmank/Dropbox/BU/NVIDIA/graphics/'
directory = 'C:/Users/pmank/Dropbox/BU/NVIDIA/graphics/'
directory_data = 'C:/Users/pmank/Dropbox/BU/NVIDIA/image_data/'
results_data = 'C:/Users/pmank/Dropbox/BU/NVIDIA/results/'
model_directory = 'C:/Users/pmank/Dropbox/BU/NVIDIA/models/model.pt'
master_data = []


# instantiate loader
loader = ImgLoader()

# check cuda
torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.get_device_name(0)

# instantiate the network
filters = 16
hidden_channels = 50
out_channels = 2
cnn = roadNet(filters, hidden_channels, out_channels, img_width, img_height)

linet = LeNet5(2)
# cnn = cnn.cuda()



###########################################
# RUN THIS CHUNK TO GET EVERYTHING FROM THE DISK
# get the image data
#loader.load_img_data_from_file(directory_data)
# get the truth values
#loader.load_truth_data_from_file(directory_data)
# get the prediction values from the network
#loader.load_pred_data_from_file(directory_data)
# get the saved model
#cnn = cnn.load(model_directory)
#########################################


# GENERATE ROAD IMAGES AND SAVE IN GRAPHICS FOLDER
truth = []
for i in range(10):
    rotation = np.random.uniform(0,60,1)
    offset = np.random.uniform(0,100,1)
    # generate the image and save the truth values
    temp = loader.generate(horizon, road_width, shoulder, offset, rotation, paint_width, directory, True)
    loader.save_file(img_index = i, path = path + "road" + str(i) + ".png").set_file_name("road" + str(i) + ".png")
    truth.append(temp)

loader.save_truth_to_disk(truth, directory_data + 'truth.npy')

# annotations_file, img_dir,transform, target_transform
# GET THE DATALOADERS
data_loaders, annotations_pd = loader.getLoaders(transform = transforms.ToTensor(), target_transform = transforms.ToTensor(), annotations_file = directory_data + 'annotations.csv', img_dir = path)

annotations_pd.head()

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 10

def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    # correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        # for each batch
        for X, y_true in data_loader:
            new_y = []
            for item in y_true:
                temp2 = []
                temp = item.strip('][').split(', ')
                for item2 in temp:
                    temp2.append(float(item2))
                new_y.append(temp2)
                y_true = torch.tensor(new_y)

            X = torch.tensor(X).float().to(device)
            y_true = torch.tensor(y_true).float().to(device)

            y_pred = model(X)
            #_, predicted_labels = torch.max(y_pred, 1)

            img_width = 1000
    
            # change to absolute value
            percent = np.sqrt(y_true - y_pred) / img_width
            # take all squares, sum them, then take the square root
            # take the sum outside of the batch loop
            # [500, 550]
            # [500, 550]

            n += y_true.size(0)
            #correct_pred += (predicted_labels == y_true).sum()

    return percent.sum() / n

def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')
X = transforms.ToTensor()

(torch.tensor[1,2,3].float())

def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    
    for i, (X, y_true) in enumerate(train_loader):
        new_y = []
        for item in y_true:
          temp2 = []
          temp = item.strip('][').split(', ')
          for item2 in temp:
            temp2.append(float(item2))
          new_y.append(temp2)
        y_true = torch.tensor(new_y)
        X = torch.tensor(X).float()
        # X = transforms.ToTensor()(X.float())
        # print("X: " + X)
        # print("Y: " + y_true)
        optimizer.zero_grad()
        
        #outputs = cnn(inputs)
#         loss = cnn_loss_fn(outputs, labels)
#         loss.backward()
#         cnn_opt.step()
#         # Sum losses
#         running_loss += loss.item()

        X = X.to(device)
        y_true = y_true.to(device)
        print(y_true)
        # Forward pass
        y_hat = model(X) 
        loss = criterion(y_hat, y_true)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print('BATCH: ' + str(i))
        print('EPOCH LOSS: ' + str(loss))
        print('RUNNING LOSS: ' + str(running_loss))
        # Backward pass
        
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
        new_y = []
        for item in y_true:
          temp2 = []
          temp = item.strip('][').split(', ')
          for item2 in temp:
            temp2.append(float(item2))
          new_y.append(temp2)
        y_true = torch.tensor(new_y)

        X = torch.tensor(X).float().to(device)
        # X = transforms.ToTensor()(X.float())
        y_true = torch.tensor(y_true).float().to(device)

        # Forward pass and record loss
        
        #print(y_true)
        y_hat = model(X) 
        #print(y_hat)
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item()
        # running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss

def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
 
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

      # plot_losses(train_losses, valid_losses)
    
    return model, optimizer, (train_losses, valid_losses)



# declare the loss function and optimizer
cnn_loss_fn = torch.nn.L1Loss()

# try lower learning rate for MSE loss
# batch norm?
cnn_opt = torch.optim.SGD(cnn.parameters(), lr=0.0001)

# TRAIN AND VALIDATE THE MODEL
# dataiter_train = next(iter(train_dataloader))
# for X in next(iter(train_dataloader)):
#   print(X)
# dataiter_test = iter(test_dataloader)
# len(train_dataloader)
#train_dataloader

dataset = data_loaders
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
# 1599 different images
dataset_size = len(dataset)
# list of number 0 - 1598
indices = list(range(dataset_size))
# calculates 20% for split (319)
split = int(np.floor(validation_split * dataset_size))
# shuffles everything and then get the indices of samples that are contained in each dataset
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

model, optimizer, losses = training_loop(cnn, cnn_loss_fn, cnn_opt, train_loader, validation_loader, epochs = 15, device = 'cpu', print_every=1)
cnn.save(model, cnn_opt, model_directory)
# plot_losses([5.622136241767202e+21, 3.281317432894323e+21, 1.9151159189031986e+21, 1.1177426831552668e+21, 6.523619552726732e+20, 3.8074627844100175e+20, 2.2221971238306457e+20, 1.2969686044315353e+20, 7.56966024349354e+19, 4.417976215686247e+19, 2.578517943186826e+19, 1.50493146670239e+19, 8.783419513463869e+18, 5.126375357578434e+18, 2.991969304317324e+18, 1.7462404835304824e+18, 1.0191800767151474e+18, 5.948369719657632e+17, 3.471721500591646e+17, 2.0262440833289558e+17],[1.0573525856549677e+21, 6.171160023031952e+20, 3.601749999700938e+20, 2.1021335588397964e+20, 1.2268946001129272e+20, 7.16067837856409e+19, 4.179277082722887e+19, 2.4392033447249076e+19, 1.4236233302388963e+19, 8.308867582924591e+18, 4.849406752997263e+18, 2.8303175975287204e+18, 1.6518938373191114e+18, 9.641151612099336e+17, 5.62698337852309e+17, 3.284146477506307e+17, 1.9167679556453987e+17, 1.1187074264312318e+17, 6.529253484743692e+16, 3.81074828869843e+16])

# plot the losses
plt.style.use('seaborn')

train_losses = np.array(losses[0]) 
valid_losses = np.array(losses[1])

fig, ax = plt.subplots(figsize = (8, 4.5))

ax.plot(train_losses, color='blue', label='Training loss') 
ax.plot(valid_losses, color='red', label='Validation loss')
ax.set(title="Loss over epochs", xlabel='Epoch',ylabel='Loss') 
ax.legend()
fig.show()





data_loaders.__getitem__(0)

loader.truth



image = loader.get_by_filename('road400.png')
image

loader.images[10]

values = list(annotations_pd[annotations_pd.name == 'road11.png'].truth)[0]
image = loader.images[11].getIMGObj()
draw = ImageDraw.Draw(image)
actual_line = [(values[1], horizon), (values[0], 750)]
draw.line(actual_line, fill = 'red')
image.show()

values

data_loaders.__getitem__(10)[1]
data_loaders.__getitem__(10)[0]
loader.truth[]


loader.truth[-1]

truth




i = 51
data_loaders.__getitem__(i-1)[0]
data_loaders.__getitem__(i-1)[1]
loader.truth[i]
    # new_y = []
    #     for item in y_true:
temp2 = []
          
temp = truth.strip('][').split(', ')
for item2 in temp:
    temp2.append(float(item2))
truth = temp2

print(truth)

y_hat = cnn(img_data.reshape((1,3,750,1000)).float())

y_hat = y_hat.tolist()[0]
actual_line = ((truth[1], horizon), (truth[0], 750))
    #print(predicted_line)
predicted_line = ((y_hat[1], horizon), (y_hat[0], 750))
print('model')
    
draw = ImageDraw.Draw(image)

draw.line(actual_line, fill = 'red')
draw.line(predicted_line, fill = 'blue')

image.show()
    



for i in range(1,len(data_loaders)):
    img_data = data_loaders.__getitem__(i-1)[0]
    truth = data_loaders.__getitem__(i-1)[1]
    image = loader.images[i].getIMGObj()
    # new_y = []
    #     for item in y_true:
    temp2 = []
          
    temp = truth.strip('][').split(', ')
    for item2 in temp:
        temp2.append(float(item2))
    truth = temp2

    print(truth)

    y_hat = cnn(img_data.reshape((1,3,750,1000)).float())

    y_hat = y_hat.tolist()[0]
    actual_line = ((truth[1], horizon), (truth[0], 750))
    #print(predicted_line)
    predicted_line = ((y_hat[1], horizon), (y_hat[0], 750))
    print('model')
    
    draw = ImageDraw.Draw(image)

    draw.line(actual_line, fill = 'red')
    draw.line(predicted_line, fill = 'blue')

    # image.show()
    image.save(results_data + 'road' + str(i) + '.png', format='png')

    # if i == 5:
    #     break


truth = loader.generate(horizon, road_width, shoulder, offset, rotation, paint_width, directory, truth_line = True)






img_data.reshape((1,3,750,1000))

# y_hat.tolist()[0|]

data_loaders.__getitem__(0)







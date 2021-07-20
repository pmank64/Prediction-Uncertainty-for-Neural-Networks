# Import required packages
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision import transforms as T
import matplotlib.pyplot as plt
from RoadImage import *
from simple_net import *
from loader import *
from torch.utils.data.sampler import SubsetRandomSampler

# set parameters for generating images
horizon = 125
road_width = 250
shoulder = 10
offset = 20
rotation = 0
paint_width = 2.5
img_width = 500
img_height = 375

rot_range = 10
offset_range = 20

results_data = "C:/Users/pmank/Dropbox/BU/NVIDIA/Part 2 - Prediction-Uncertainty-for-Neural-Networks/results"
path = 'C:/Users/pmank/Dropbox/BU/NVIDIA/Part 2 - Prediction-Uncertainty-for-Neural-Networks/graphics/'
directory = 'C:/Users/pmank/Dropbox/BU/NVIDIA/Part 2 - Prediction-Uncertainty-for-Neural-Networks/graphics/'
directory_data = 'C:/Users/pmank/Dropbox/BU/NVIDIA/Part 2 - Prediction-Uncertainty-for-Neural-Networks/image_data/'
results_data = 'C:/Users/pmank/Dropbox/BU/NVIDIA/Part 2 - Prediction-Uncertainty-for-Neural-Networks/results/'
model_directory = 'C:/Users/pmank/Dropbox/BU/NVIDIA/Part 2 - Prediction-Uncertainty-for-Neural-Networks/model.pt'


learning_rate = 3e-07
num_epochs = 100
num_epochs_to_print = 1

# fix random seeds so that results are reproduceable from run to run
torch.manual_seed(0)
np.random.seed(0)


##GENERATE IMAGES#####
truth = []
images = []
num_images = 2500
start_valid = num_images / 2
if start_valid > num_images -1:
    start_valid = num_images -1
start_valid =int(start_valid)

# instantiate a new loader object
loader = ImgLoader()

# zero out everything in the loader
loader.reset()

for i in range(0, num_images):
    rotation = np.random.uniform(0,rot_range,1) 
    # offset should be on a range from a negative to positive
    offset = np.random.uniform(-offset_range/2, offset_range/2, 1)
    # generate the image with the loader
    # (the loader saves the truth value internally)
    loader.generate(horizon, road_width, shoulder, offset, rotation, paint_width, "", False)

# add noise to image data - this function iterates through the road image objects in the loader and adds noise, saving the 
# numerical noisy image data within the loader
loader.make_noise(100)
    
   

# get the data loaders and annotations dataframe from the loader
data_loaders, annotations_pd = loader.getLoaders(transform = transforms.ToTensor(), target_transform = transforms.ToTensor(), annotations_file = directory_data + 'annotations.csv', img_dir = path)

dataset = data_loaders
batch_size = 10
validation_split = .3
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
# list of number 0 --> num_images
indices = list(range(dataset_size))
# calculates the split
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



# instantiate the model
model = simpleNet()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# set these lists to keep track of the losses in the train and test datasets
losses_train = []
losses_valid = []

# training 
def forward_pass(model):
    running_loss = 0
    model.train()
    for i, (X, y_true) in enumerate(train_loader):
        loss_fn = torch.nn.MSELoss()
        y_pred = model(X.float())
        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = np.abs(running_loss / (i+1))
    losses_train.append(epoch_loss)
    return loss, y_pred

# validation
def valid(model):
    running_loss = 0
    model.eval()
    for i, (X, y_true) in enumerate(validation_loader):
        loss_fn = torch.nn.MSELoss()
        y_pred = model(X.float())
        loss = loss_fn(y_pred, y_true)
        running_loss = running_loss + loss.item()
    epoch_loss = np.abs(running_loss / (i+1))
    losses_valid.append(epoch_loss)
    return loss, y_pred


## MAIN TRAINING LOOP
for t in range(num_epochs):  
    train_loss, train_pred  = forward_pass(model)
    with torch.no_grad():
        valid_loss, valid_pred  = valid(model)
    # report results after every "num_epochs_to_print" epochs
    if t % num_epochs_to_print == 0:
        print("t = ", t, "   train loss = ", np.sqrt(train_loss.item()),  "   valid loss = ", np.sqrt(valid_loss.item()))
print("done")
######


##GENERATE THE PLOT############
train_losses = np.sqrt(np.array(losses_train))
valid_losses = np.sqrt(np.array(losses_valid))

fig, ax = plt.subplots(figsize = (8, 4.5))
ax.plot(train_losses, color='blue', label='Training loss') 
ax.plot(valid_losses, color='red', label='Validation loss')
ax.set(title="Loss over epochs", xlabel='Epoch',ylabel='Loss') 
ax.legend()
plt.show()
###########




# ##VIEW AN IMAGE########
# change img_idx to view a different image with the truth and predicted lines
img_idx = 20
# demo_array = np.moveaxis(images_valid[img_idx].numpy(), 0, -1)
demo_array = np.moveaxis(data_loaders.__getitem__(img_idx)[0].numpy(), 0, -1)
demo_array = torch.tensor(demo_array).numpy()

im = Image.fromarray(demo_array.astype(np.uint8))

y_hat = model(torch.tensor(data_loaders.__getitem__(img_idx)[0].numpy()).reshape((1,3,375,500)).float())
y_hat = y_hat.tolist()[0]
# actual_line = ((data_loaders.__getitem__(img_idx)[1][0], horizon), (data_loaders.__getitem__(img_idx)[1][1], 750))
# predicted_line = ((y_hat[0], horizon), (y_hat[1], 750))


draw = ImageDraw.Draw(im)
# draw.line(actual_line, fill = 'red')
# draw.line(predicted_line, fill ='blue')
im.show()
###########



# VIEW ERROR ON AN INDIVIDUAL IMAGE
img_idx = 20
predicted = model(torch.tensor(data_loaders.__getitem__(img_idx)[0].numpy()).reshape((1,3,375,500)).float())[0][0]
actual = data_loaders.__getitem__(img_idx)[1][0]
print("Predicted: " + str(predicted))
print("Actual: " + str(actual))
print("Error: " + str(float(actual - predicted)))


import torch.nn  as  nn
import torch
from torchvision import transforms, datasets
import torch.optim as  optim
import matplotlib.pyplot as  plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import os

# GLOBAL CONTANTS ( global constants should always be written with caps. )
INPUT_DIM = (256, 256) 
CHANNEL_DIM = 3 # RGB
CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = "group_8.pth"
EPOCHS = 2

# We chose to use 3 channels (RGB) to preserve potential information 
# that might get lost with 1 channel (grayscale). Due to the dataset
# not being to large, we did it to ensure that color-based distinctions and nuances, 
# which might be relevant even in seemingly monochromatic datasets, are not overlooked. 


# Define model
# A Convolutional Neural Network is chosen due to its strengths in image recognition and classification. 
# As our purpose is to distinguish the images of the healthy patients from those with pneumonia, we find it to be the best option.
class group_8(nn.Module):
    def __init__(self):
        super(group_8, self).__init__()

        ## CONV LAYER
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0) # torch.Size([64, 16, 254, 254])
        self.pool = nn.MaxPool2d(kernel_size=2) # torch.Size([64, 16, 127, 127])

# We have chosen to make 3 fully connected layers in a funnel structure.
        self.fc1 = nn.Linear(in_features=16*127*127, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=2)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x) # After conv1 - x.shape : torch.Size([64, 16, 254, 254])
        # print(f"After conv1 - x.shape : {x.shape}")
        x = self.pool(x) # After maxpool - x.shape : torch.Size([64, 16, 127, 127])
        # print(f"After maxpool - x.shape : {x.shape}")
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        ## DROPOUT HERE
        ## RIDGE REGRESSION?
        x = self.relu(self.fc2(x))
        ## DROPOUT HERE
        ## RIDGEREGRESSION?
        x = self.sigmoid(self.fc3(x))
        return x
    
# CNN Architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)  # Two classes: normal and pneumonia

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv3(x))
        x = self.maxpool(x)
        x = torch.relu(self.conv4(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
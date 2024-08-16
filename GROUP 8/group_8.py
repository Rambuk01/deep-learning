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
input_dim = (256, 256)
channel_dim = 3
CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = "g8.pth"
EPOCHS = 10

# Define model
class group_8(nn.Module):
    def __init__(self):
        super(group_8, self).__init__()

        # Regarding the parameters we set the in_channels to 3 for best fit on RGB (Red,green blue) images
        # We are starting with a low degree of out_channels and increase this in our next layers
        # Our Kernel size is set to 3x3 for optimal fit that can capture the finer details in the 
        # The stride level is set to 1 which means the kernel moves with one pixel with each iterations, so with higher stride it would skip an iteration
        # We set padding to 1 to keep the size after the convulation. It secures that the spatial dimensatiosn remains consistent

        ## CONV LAYER
        self.conv1 = nn.Conv2d(in_channels=channel_dim, out_channels=16, kernel_size=3, stride=1, padding=1) # torch.Size([64, 16, 254, 254])

        #We are using maxpooling to reduce the spatial dimensions in order to reduce computational load, where we set our kernel size to 
        # 2x2 which should be appropriate for this type of images
        self.pool = nn.MaxPool2d(kernel_size=2) # torch.Size([64, 16, 127, 127])

        self.dropout = nn.Dropout(0.3)

        # The output size is 2 since we have two scenarios with disease or without disease
        # The hidden layer size we have set to respectively 512 followed by 256 which should be enough to capture the complex patterns

        self.fc1 = nn.Linear(in_features=16 * 128 * 128, out_features=512)  # Adjusted in_features and increased out_features
        self.fc2 = nn.Linear(in_features=512, out_features=256)            # Intermediate layer size
        self.fc3 = nn.Linear(in_features=256, out_features=2)                   


        # RELU is needed in CNNs in order to introducde non-linearity, which is needed for the netwrok to learn more complex patterns
        # The sigmoid function is used for binary classifications, which is strongly related to the logistic regression (again, for 2 outcomes)
        # Softmas is not really clear while we use this for binary classification, since its an activation for multi class problems

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


        # The forward method which  we are using here represents a common architecture that is suitable for classifications task
        # It takes in the parameter tunings which we have defined above and set to an appropriate amount related classifying pneumonia in lungs
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x) # After conv1 - x.shape : torch.Size([64, 16, 254, 254])
        # print(f"After conv1 - x.shape : {x.shape}")
        x = self.pool(x) # After maxpool - x.shape : torch.Size([64, 16, 127, 127])
        # print(f"After maxpool - x.shape : {x.shape}")
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))  ## RIDGE REGRESSION HERE
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

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
CHANNEL_DIM = 3
CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = "best_cnn_model.pth"

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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) # torch.Size([64, 16, 254, 254])

#We are using maxpooling to reduce the spatial dimensions in order to reduce computational load, where we set our kernel size to 
# 2x2 which should be appropriate for this type of images
        self.pool = nn.MaxPool2d(kernel_size=2) # torch.Size([64, 16, 127, 127])


# The output size is 2 since we have two scenarios with disease or without disease
# The hidden layer size we have set to respectively 512 followed by 256 which should be enough to capture the complex patterns

        self.fc1 = nn.Linear(in_features=16 * 15 * 15, out_features=512)  # Adjusted in_features and increased out_features
        self.fc2 = nn.Linear(in_features=512, out_features=256)            # Intermediate layer size
        self.fc3 = nn.Linear(in_features=256, out_features=2)                   


# RELU is needed in CNNs in order to introducde non-linearity, which is needed for the netwrok to learn more complex patterns
# The sigmoid function is used for binary classifications, which is strongly related to the logistic regression (again, for 2 outcomes)
# Softmas is not really clear while we use this for binary classification, since its an activation for multi class problems

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax2d()
#        self.sigmoid = nn.Sigmoid()


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
        ## DROPOUT HERE
        ## RIDGE REGRESSION?
        x = self.relu(self.fc2(x))
        ## DROPOUT HERE
        ## RIDGEREGRESSION?
        x = self.sigmoid(self.fc3(x))
        return x
    
# CNN Architecture

# The chanels continiously increase (eg. 32 to 64 to 128 etc), since this is common practice with image recognition.
# It start with the input image, whereafter it extract more and more low level features (smaller detatils in the picture) layer after layer. 
# For our maxpooling we have chosen the same kernel size and stride as mentioned before
# Dropput is a regulatization method, which is used to prevent overfitting (to downregulate the rate of overfitting). Specificially, dropout 
# is "dropping out" a share of neurons 
 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 16 * 16, 512)  
        self.fc2 = nn.Linear(512, 2)  # Two classes: normal and pneumonia


# Max pooling (downsampling) is used to make the model invariant to translation (small changes in the input image). 
# This means that the model can recognize the object/feature of interest (for instance the disease) regardless of the how the
# visual pattern of the disease is present in the picture. Max pooling means taking the "highest number", in constrast to average pooling.
# The higher the "number", the stronger the feature.
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.relu(self.conv5(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
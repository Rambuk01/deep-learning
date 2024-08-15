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

# Define model
class group_8(nn.Module):
    def __init__(self):
        super(group_8, self).__init__()
        self.fc1 = nn.Linear(in_features=256*256*3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x
    

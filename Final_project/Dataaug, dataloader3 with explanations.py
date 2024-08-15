# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:37:56 2024

@author: Geo V. Søndergaard
"""

import os                    # For directory and file management
import glob                  # For file pattern matching
from PIL import Image        # For image handling
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# 1. **Custom Dataset Class Implementation**:
# The CustomDataset class is created to manage the loading of images and their labels, 
# as well as applying transformations. Changes made include:
# - Improved initialization with clear parameters.
# - Read images in a method that provides flexibility.
# - Added standard transformations for normalization.

class CustomDataset(Dataset):
    def __init__(self, img_size, class_names, path=None, transformations=None, num_per_class: int = -1):
        self.img_size = img_size  # Size to which images will be resized (tuple).
        self.path = path           # Where the dataset is located.
        self.num_per_class = num_per_class  # Optional limit on the number of images for each class.
        self.class_names = class_names  # A mapping of classes to their corresponding labels.
        self.transforms = transformations  # Data transformations for images.
        self.data = []  # List to hold loaded images.
        self.labels = []  # List to hold labels corresponding to the images.

        if path:  # If a path is provided, load images
            self.read_images()  # Call method to read images from the specified directory.

        # Standard transforms for normalization. Changes include using the ImageNet mean and std for normalization
        # as it often improves convergence in CNN architectures when trained on similar datasets.
        self.standard_transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensor format.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet statistics.
        ])

    # 2. **Image Reading Method**:
    # This method manages image loading from the directory specified in the path.
    # Changes made include:
    # - Utilizing glob to find all images in specified classes.
    # - Properly handling the loading of images while ensuring their corresponding labels are collected.

    def read_images(self):
        for id, class_name in self.class_names.items():  # Iterate through classes.
            print(f'Loading images from class: {id} : {class_name}')  # Print loading status.
            img_paths = glob.glob(f'{self.path}{class_name}/*.jpg')  # Get all image paths for the class.
            if self.num_per_class > 0:  # Limit the number of images per class if specified.
                img_paths = img_paths[:self.num_per_class]
            self.labels.extend([id] * len(img_paths))  # Extend label list.
            for filename in img_paths:  # Load each image found.
                img = Image.open(filename).convert('RGB')  # Open image and convert to RGB.
                img = img.resize(self.img_size)  # Resize the image to the defined img_size.
                self.data.append(img)  # Append the image to the data list.

    # 3. **Dataset Length and Item Retrieval**:
    # Implements the built-in methods needed for the Dataset class.
    # Change considerations:
    # - Ensure correct data retrieval through indexing, which is essential for DataLoader.

    def __len__(self):
        return len(self.data)  # Return the total number of samples.

    def __getitem__(self, idx):
        img = self.data[idx]  # Get the image at the specified index.
        label = self.labels[idx]  # Get the corresponding label.

        # Apply transformations, if specified; otherwise use standard transformations.
        if self.transforms:
            img = self.transforms(img)
        else:
            img = self.standard_transforms(img)  # Use the default normalization if no transforms provided.

        label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor.

        return img, label  # Return the processed image and label.


# 4. **Define Transformations for Training and Validation**:
# Establishes the data augmentations and normalization for the training and validation datasets.
# Changes made emphasize:
# - Extensive augmentations for the training transformation to enhance model robustness.
# - Simpler, consistent normalization for the validation dataset to avoid data leakage.

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally.
    transforms.RandomVerticalFlip(),       # Randomly flip the image vertically.
    transforms.RandomRotation(degrees=30), # Random rotation within 30 degrees.
    transforms.RandomAffine(degrees=30, translate=(0.1, 0.1)),  # Apply random affine transformations.
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Random color adjustments.
    transforms.Resize((256, 256)),        # Resize images to ensure consistency in size.
    transforms.ToTensor(),                 # Convert images to a tensor.


# Validation transformations should be simpler, focusing solely on normalization.
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Consistent size for validation images.
    transforms.ToTensor(),            # Convert to tensor.
    

# 5. **Load Data**:
# Set paths for datasets and create instances of the dataset class.
# Changes made include characterized paths to ensure full configurability and flexibility for dataset management.

train_path = "./data/train/"  # Path to training images.
val_path = "./data/val/"       # Path to validation images.

# Fetch class names from the training dataset directory.
class_names = [name[len(train_path):] for name in glob.glob(f'{train_path}*')]
class_names = dict(zip(range(len(class_names)), class_names))  # Map class indices to labels.

# Create an instance of the CustomDataset with specified parameters.
full_dataset = CustomDataset(img_size=(256, 256), path=train_path, class_names=class_names,
                             transformations=train_transform, num_per_class=-1)

# 6. **Dataset Splitting**:
# Randomly split the dataset into training and validation sets and apply the proper transformation.
# This section emphasizes clear dataset integrity and validation.

train_size = int(0.8 * len(full_dataset))  # Use 80% of the dataset for training.
val_size = len(full_dataset) - train_size   # Remaining portion for validation.

# Split the dataset into a training and validation set.
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.transforms = val_transform  # Assign the validation dataset transforms.


# 7. **Data Loaders**:
# Create data loaders for training and validation datasets, optimizing the batching process.

batch_size = 64  # Define the number of samples per batch for training and validation.
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle training data for randomness.
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)   # No shuffle for validation data to maintain order.

# 8. **Define Model (Without Pretraining)**:
# Initialize the ResNet-50 model from scratch
resnet_model = torchvision.models.resnet50(weights=None)  # Change made here: Load model from scratch

# Modify the fully connected layer to match the number of classes
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, len(class_names))  # Adjust output layer
resnet_model.to(device)  # Move model to the appropriate device (CPU or GPU)

# Following this, you would typically define loss function and optimizer, e.g.:
loss_fn = nn.CrossEntropyLoss()  # Using CrossEntropyLoss for multi-class classification
optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)  # Example optimizer


import torch.nn  as  nn
import torch
from torchvision import transforms, datasets
import torch.optim as  optim
import matplotlib.pyplot as  plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from shutil import move
import random
import torch.nn.functional as F
from group_8 import *

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = ['normal', 'pneumonia']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        # Collect all image paths and corresponding labels
        for cls_name in self.classes:
            cls_dir = os.path.join(root, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB
        label = self.labels[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label #, os.path.basename(img_path)

INPUT_DIM = (256, 256)

train_transform = transforms.Compose([
    transforms.Resize(INPUT_DIM),  # Resize all images to 256x256
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Reduced transformations suitable for medical images
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Adjust brightness and contrast slightly
    transforms.ToTensor(),
])
test_val_transform = transforms.Compose([
    transforms.Resize(INPUT_DIM),  # Resize all images to 256x256
    transforms.ToTensor(),
])

# LOAD THE DATASETS
train_dataset = CustomDataset(root=f"/{CURRENT_FILE_DIRECTORY}/organized_data/training", transform=train_transform)
val_dataset = CustomDataset(root=f"/{CURRENT_FILE_DIRECTORY}/organized_data/validation", transform=test_val_transform)
test_dataset = CustomDataset(root=f"/{CURRENT_FILE_DIRECTORY}/organized_data/testing", transform=test_val_transform)

# DATA LOADERS
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

# Training and Evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, num_epochs=25):
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

model = group_8()
model.load_state_dict(torch.load(MODEL_PATH))

train(model, num_epochs=1)
val_accuracy = evaluate(model, val_loader)
test_accuracy = evaluate(model, test_loader)
print(f"Validation Accuracy: {val_accuracy:.2f}%")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Save the best model
torch.save(model.state_dict(), MODEL_PATH)

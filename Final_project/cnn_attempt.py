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
from pathlib import Path


# Functions thats trains the model using our dataloader. 
def train(model, loader, criterion, optimizer):

    model.train()
    running_loss = 0.0

    # Training loop
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # Clears any previous gradients.
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # Performing backpropagation to compute gradients.
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    return avg_train_loss

# Evaluation function
def evaluate(model, loader):
    model.eval()
    val_loss = 0.0  # Initializes the validation loss
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)

            loss = criterion(outputs, labels)  # Computing the validation loss
            val_loss += loss.item()  # Accumulating the validation loss

            _, predicted = torch.max(outputs.data, 1) 
            # Determines the predicted class from the highest score.

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    avg_val_loss = val_loss / len(loader)  # Calculates the average validation loss
    accuracy = 100 * correct / total

    return accuracy, avg_val_loss

# Using Pytorchs ´CustomDataset´ to work with the image classification task.
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
        image = Image.open(img_path).convert('RGB')  # Converting to RGB
        label = self.labels[idx]

        # Applying the transformations
        if self.transform:
            image = self.transform(image)

        return image, label

# Making individual transformations on train and test/validation data. 
train_transform = transforms.Compose([
    transforms.Resize(INPUT_DIM),  # Resize all images to 256x256 
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # For slightly altered medical images
    transforms.RandomHorizontalFlip(),  # Flip is useful since pneumonia might appear similary on either lung
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Adjust brightness and contrast slightly, since medical images ofent have it standardized.
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


# Training and Evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = group_8()
model.to(device)

## LOAD THE TRAINED MODEL STATE ##
model_file = Path(MODEL_PATH)
if model_file.is_file():
    print("LOADING THE TRAINED MODEL STATE")
    model.load_state_dict(torch.load(MODEL_PATH))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# Loss and accuracy lists
train_losses = []
val_losses = []
val_accuracies = []
prior_val_accuracy = 0;

collector = [];

### EPOCH TRAINING LOOP ###
for epoch in range(EPOCHS):
    # Train model
    avg_train_loss = train(model, train_loader, criterion, optimizer)

    # Evaluate model on validation dataset
    val_accuracy, avg_val_loss = evaluate(model, val_loader)

    ####### SAVE MODEL IF IT IS BETTER ########
    if val_accuracy > prior_val_accuracy:
        torch.save(model.state_dict(), f"{MODEL_PATH}")
        print(f"Model improved: New Accuracy: {val_accuracy}, Old Accuracy: {prior_val_accuracy}")
        print(f"Saving new model improved: New Accuracy: {val_accuracy}")
        prior_val_accuracy = val_accuracy;

    # Save accuracy and loss data
    train_losses.append(avg_train_loss) # Saves train_loss
    val_losses.append(avg_val_loss)  # Stores average validation loss
    val_accuracies.append(val_accuracy)  # Calculates accuracy as a percentage


    print(f'Epoch [{epoch+1}/{EPOCHS}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')


# Test the final model on the test dataset
test_accuracy, _ = evaluate(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%")

######## PLOT ########
# Plot training and validation loss, along with accuracy
plt.figure(figsize=(12, 6))

train_loss_line, = plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
val_loss_line, = plt.plot(val_losses, label='Validation Loss', color='green', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss', color='blue')
plt.tick_params(axis='y', labelcolor='blue')
ax2 = plt.gca().twinx()

accuracy_line, = ax2.plot(val_accuracies, label='Validation Accuracy', color='orange', marker='s')
ax2.set_ylabel('Accuracy (%)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
lines = [train_loss_line, val_loss_line, accuracy_line]
labels = [line.get_label() for line in lines]
plt.title('Training Loss, Validation Loss, and Accuracy')
plt.legend(lines, labels, loc='upper left')
plt.savefig(f'{CURRENT_FILE_DIRECTORY}/group_8.png')
plt.show()

import torch
import torch.nn  as  nn  
import torch.optim as  optim
import torch.utils
import torchvision
from    torchvision import datasets,  transforms
import seaborn as  sns   
import matplotlib.pyplot as  plt
import sys
import numpy as np
import os
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing the images.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.image_dir = image_dir
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(image_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_path = os.path.join(image_dir, cls_name)
            # Check if cls_path is a directory
            if not os.path.isdir(cls_path):
                continue

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                # Skip non-image files (you can add more extensions if needed)
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    continue

                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

        self.transform = transform
        self.label_adjustment_needed = min(self.labels) == 1


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Retrieves an image and its corresponding label."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Subtract 1 from the label if the dataset is the test dataset and labels start from 1
        # Adjust label if necessary
        if self.label_adjustment_needed:
            label -= 1

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize all images to 256x256
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Create instances of the dataset
train_dataset = CustomDataset(image_dir='exercise05/train/', transform=transform)
test_dataset = CustomDataset(image_dir='exercise05/test/', transform=transform)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# Check the dataset
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Example: Iterate through the training loader
for images, labels in train_loader:
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels shape: {labels.shape}")
    break

# For newer versions of torchvision
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)


# change final layer
model.fc = nn.Linear(model.fc.in_features, 16)

for param in model.parameters():
    param.requires_grad = False # Freeze weights of the model

for param in model.fc.parameters():
    param.requires_grad = True # Unfreeze last layer weights, so we can train these.


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training settings
num_epochs = 5

train_losses = []
val_losses = []
test_accuracies = []

## EPOCH ## 
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss for the epoch
    
    unique_labels = set() # DEBUGGING!!

    ## TRAINING LOOP ##
    for inputs, targets in train_loader:
        unique_labels.update(targets.numpy())

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass: compute the model output
        loss = criterion(outputs, targets)  # Compute the loss

        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Optimize the weights based on gradients
        running_loss += loss.item()  # Accumulate the loss for the batch
        
    print(f"Unique labels in the dataset: {sorted(unique_labels)}")
    # Track average training loss
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    ## EVALUATE ##
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0  # Initialize validation loss
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, targets in test_loader:

            outputs = model(inputs)  # Forward pass: compute the model output
            loss = criterion(outputs, targets)  # Compute validation loss
            val_loss += loss.item()  # Accumulate validation loss
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)  # Get the index of the maximum log-probability
            total += targets.size(0)  # Update total predictions count
            correct += (predicted == targets).sum().item()  # Update correct predictions count

    avg_val_loss = val_loss / len(test_loader)  # Calculate average validation loss
    val_losses.append(avg_val_loss)  # Store average validation loss
    test_accuracy = 100 * correct / total  # Calculate accuracy as a percentage
    test_accuracies.append(test_accuracy)  # Store accuracy
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
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





# Assume the test_loader is already defined
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

test_dataset = CustomDataset(image_dir='exercise05/test/', transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)




# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the ResNet-50 model (you can reinstantiate it as shown in the image)
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 16)

# Load the trained weights
model.load_state_dict(torch.load("exercise05/model.pth"))
model.to(device)

# Set the model to evaluation mode
model.eval()

# Initialize counters for correct predictions and total samples
correct = 0
total = 0

# Disable gradient calculation for evaluation
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass: compute predicted outputs
        outputs = model(inputs)
        
        # Get the predicted class with the highest score
        _, predicted = torch.max(outputs.data, 1)
        
        # Update total and correct counts
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        

# Calculate the accuracy
accuracy = 100 * correct / total
print(f'Accuracy of the model on the test dataset: {accuracy:.2f}%')

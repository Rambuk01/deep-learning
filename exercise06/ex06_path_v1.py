import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # Initial fully connected layer
        self.fc1 = nn.Linear(28 * 28, 28 * 28)  # Input layer
        
        # Forking paths with convolutional layers
        # Path 1
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # 1 input channel, 16 output channels
        self.conv1_2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 16 input channels, 32 output channels
        
        # Path 2
        self.conv2_1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)  # 1 input channel, 16 output channels
        self.conv2_2 = nn.Conv2d(16, 32, kernel_size=5, padding=2) # 16 input channels, 32 output channels
        
        # Fully connected layers after concatenation
        self.fc2 = nn.Linear(32 * 7 * 7 * 2, 64)  # Output size after concatenation
        self.fc3 = nn.Linear(64, 10)  # Final output layer for 10 classes (MNIST)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax along the class dimension (dim=1)
        
    def forward(self, x):
        # Flatten the input
        x = torch.flatten(x, 1)
        
        # Initial fully connected layer
        x = F.relu(self.fc1(x))

        # Reshape to image format for convolutional layers
        x = x.view(256, 1, 28, 28)
        
        # Forked path 1
        x1 = F.relu(self.conv1_1(x))
        x1 = F.max_pool2d(F.relu(self.conv1_2(x1)), 2)  # Max pooling

        # Forked path 2
        x2 = F.relu(self.conv2_1(x))
        x2 = F.max_pool2d(F.relu(self.conv2_2(x2)), 2)  # Max pooling

        # Print shapes to debug
        print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")
        
        # Concatenate the outputs from both paths
        x_cat = torch.cat((x1, x2), dim=1)  # Concatenate along the channel dimension
        x_cat = F.max_pool2d(x_cat, 2)
        # Flatten the concatenated output for the fully connected layers
        print(f"x_cat shape before flattening: {x_cat.shape}")
        x_cat = x_cat.view(x_cat.size(0), -1)  # Use dynamic batch size
        
        # Check the size after flattening
        #print(f"x_cat shape after flattening: {x_cat.shape}")
        
        # Final fully connected layers
        x = F.relu(self.fc2(x_cat))
        x = self.softmax(self.fc3(x))  # Apply softmax to the final output
        
        return x



model = MyNetwork()
print(model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01) #momentum=0.9



# Training settings
num_epochs = 5

train_losses = []
val_losses = []
test_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss for the epoch
    
    # Training loop
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass: compute the model output
        loss = criterion(outputs, targets)  # Compute the loss

        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Optimize the weights based on gradients
        running_loss += loss.item()  # Accumulate the loss for the batch

    # Track average training loss
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluate on test set
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

# Plot training and validation loss, along with accuracy
plt.figure(figsize=(12, 6))

train_loss_line, = plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
val_loss_line, = plt.plot(val_losses, label='Validation Loss', color='green', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss', color='blue')
plt.tick_params(axis='y', labelcolor='blue')
ax2 = plt.gca().twinx()
accuracy_line, = ax2.plot(test_accuracies, label='Validation Accuracy', color='orange', marker='s')
ax2.set_ylabel('Accuracy (%)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
lines = [train_loss_line, val_loss_line, accuracy_line]
labels = [line.get_label() for line in lines]
plt.title('Training Loss, Validation Loss, and Accuracy')
plt.legend(lines, labels, loc='upper left')
plt.savefig(f'exercise04/DEEP.png')
# plt.show()
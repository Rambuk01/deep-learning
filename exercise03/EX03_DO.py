import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import seaborn as sns
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True,
transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True,
transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
batch_size=256, shuffle=False)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.dropout = nn.Dropout(0.3)  # 2.a add dropout layer

        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x) # add dropout layer
        x = torch.relu(self.fc2(x))
        x = self.dropout(x) # add dropout layer
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Regularization parameters
    
model = NeuralNet()
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #momentum=0.9



# Training settings
num_epochs = 100

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
        l1_norm_fc2 = torch.norm(model.fc2.weight, p=2)  # Calculate L2 norm of the weights in the 2nd layer
        loss += 0.001 * l1_norm_fc2  # Add L1 regularization to the loss

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
plt.savefig(f'exercise03/asd2.png')
# plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class MNISTNetwork(nn.Module): # Define the MNISTNetwork class that inherits from nn.Module
    def __init__(self): # Initialize the neural network (itself)
        super(MNISTNetwork, self).__init__() # Call the constructor of the parent class (nn.Module)
        self.fc1 = nn.Linear(28*28, 256)  # First fully connected layer (identifiy  the widest variety on features. note: started with 512 but it was overfitting so lowered it )
        self.fc2 = nn.Linear(256, 128)    # Second fully connected layer (focus on more complex features found in previous layer)
        self.fc3 = nn.Linear(128, 64)    # Third fully connected layer (focus on more complex features found in previous layer)
        self.fc4 = nn.Linear(64, 10)     # Output layer

    def forward(self, x): # Define the forward pass of the network
        x = x.view(-1, 28*28)  # Reshape the input tensor to be 2D (batch_size, 28*28)
        x = torch.relu(self.fc1(x)) # Apply the ReLU activation function after layer 1
        x = torch.relu(self.fc2(x)) # Apply the ReLU activation function after layer 2
        x = torch.relu(self.fc3(x)) # Apply the ReLU activation function after layer 3
        x = self.fc4(x)  # Output layer (no activation for logits)
        return x # Return the output of the network

# Instantiate the model
model = MNISTNetwork() # Create an instance of the MNISTNetwork class

criterion = nn.CrossEntropyLoss()  # Define the loss function (cross-entropy loss for multi-class classification) 
optimizer = optim.Adam(model.parameters(), lr=0.001 )  # Define the optimizer (Adam) and learning rate (0.001)

num_epochs = 10 # Set the number of training epochs
train_losses = [] # Initialize a list
val_losses = [] # Initialize a list
accuracies = [] # Initialize a list

# Training loop
for epoch in range(num_epochs): # Loop over the number of epochs
    model.train() # Set the model to training mode
    running_loss = 0.0 # Initialize running loss for the epoch
    for images, labels in train_loader: # Loop over the training dataset
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass: compute the model output
        loss = criterion(outputs, labels) # Compute the loss between outputs and true labels
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Optimize the weights based on gradients
        running_loss += loss.item() # Accumulate the loss for the batch
    avg_train_loss = running_loss / len(train_loader) # Calculate average training loss for the epoch
    train_losses.append(avg_train_loss) # Store average training loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

    # Validation phase
    model.eval() # Set the model to evaluation mode
    val_loss = 0.0 # Initialize
    correct = 0 # Initialize
    total = 0 # Initialize

    with torch.no_grad(): # Disable gradient calculation for validation
        for images, labels in test_loader: # Loop over the test dataset
            outputs = model(images) # Forward pass: compute the model output
            loss = criterion(outputs, labels)  # Compute validation loss
            val_loss += loss.item()  # Accumulate validation loss

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1) # Get the index of the maximum log-probability
            total += labels.size(0) # Update total predictions count
            correct += (predicted == labels).sum().item() # Update correct predictions count

    avg_val_loss = val_loss / len(test_loader) # Calculate average validation loss
    val_losses.append(avg_val_loss) # Store average validation loss
    accuracy = 100 * correct / total # Calculate accuracy as a percentage
    accuracies.append(accuracy) # Store accuracy
    print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Save the model
save_path = r'C:/Users/Aiga/Desktop/SDU/Session 2/mnist_model.pth'
torch.save(model.state_dict(), save_path)

# Plotting training and validation loss
plt.figure(figsize=(10, 5)) # Create a figure for plotting
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.show()

# Reporting accuracy
final_accuracy = accuracies[-1] # Get the final accuracy from the accuracies list
if final_accuracy >= 90:  # Assuming 90% accuracy is satisfactory
    print(f'Final Accuracy: {final_accuracy:.2f}% - This is satisfactory.')
else:
    print(f'Final Accuracy: {final_accuracy:.2f}% - This is not satisfactory, try again.')

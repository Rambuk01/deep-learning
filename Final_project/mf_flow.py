from group_8 import *
## MARIO ##
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


transform = transforms.Compose([
    transforms.Resize(INPUT_DIM),  # Resize all images to 256x256
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Rotation (degrees): The image will be randomly rotated within the specified degree range. Translation (translate): The image will be randomly shifted horizontally and/or vertically by up to 10% of its dimensions. Scaling (scale): The image will be randomly resized, either zoomed in or out within the specified scaling range.
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)), ## DONT INCLUDE THIS, UNLESS YOU KNOW EXACTLY WHAT IT DOES
])

# LOAD THE DATASETS
train_dataset = CustomDataset(root=f"/{CURRENT_FILE_DIRECTORY}/organized_data/training", transform=transform)
val_dataset = CustomDataset(root=f"/{CURRENT_FILE_DIRECTORY}/organized_data/validation", transform=transform)
test_dataset = CustomDataset(root=f"/{CURRENT_FILE_DIRECTORY}/organized_data/testing", transform=transform)

# DATA LOADERS
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = 'cpu' # OVERWRITING DEVICE. A FEW PROBLEMS USING MPS.
print(f"Using {device} device")





############ START INSPECT ... THE TRAIN DATA SET. TO SEE IF IT WORKS AS INTENDED! CHECK IMAGE DIMENSIONS, DATA LENGTH AND SHOW FIRST IMAGE.
"""
print(f"Number of samples in the training dataset: {len(train_dataset)}")
image, label = train_dataset[-2]  # Access the first sample
print(f"Label: {label}")
print(f"Image size: {image.size()}")  # If the image is a tensor, use .size(), for PIL image, use .size
import matplotlib.pyplot as plt
import numpy as np

# Convert the tensor image back to numpy for visualization (assuming it's normalized)
image_np = image.numpy().transpose((1, 2, 0))  # Convert to HWC format
image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Un-normalize
image_np = np.clip(image_np, 0, 1)  # Clip values to be between 0 and 1

plt.imshow(image_np)
plt.title(f"Label: {label}")
plt.show()
sys.exit()
"""
############ END INSPECT


############ INSPECT DATA LOADER
"""
for images, labels in train_loader:
    # images: a batch of images with shape (batch_size, 3, 224, 224)
    # labels: a batch of labels with shape (batch_size), containing 0s and 1s
    print(images.shape)
    print(labels)
    break  # Just to print the first batch
"""
############ END INSPECT DATA LOADER


## INITIATE MODEL
model = group_8().to(device)
print(model)

## LOSS FUNCTION AND OPTIMIZER
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
    
    ## TRAINING LOOP ##
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)  # Forward pass: compute the model output
        loss = criterion(outputs, targets)  # Compute the loss

        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Optimize the weights based on gradients
        running_loss += loss.item()  # Accumulate the loss for the batch
        
    # Track average training loss
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    ## EVALUATE ##
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0  # Initialize validation loss
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

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


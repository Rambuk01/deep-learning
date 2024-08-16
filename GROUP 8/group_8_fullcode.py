import torch.nn  as  nn
import torch
from torchvision import transforms
import torch.optim as  optim
import matplotlib.pyplot as  plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from shutil import move
import random
import torch.nn.functional as F
from pathlib import Path
"""
########## NOTE ##########
# We have all been working on every task, and put effort into understand all the code.

Group 8 Memebers:
- Alberte Krogh Hansen
- Aiga Kalneja
- Anton Rindom
- Geo Vittrop Søndergaard
- Maria Post Hofmann
- Mario Hernández Festersen

########## TASK 1 ##########
# Mainly Written by Aiga, Maria

########## TASK 2 ##########
# Mainly Written by Anton, Mario, Alberte, Geo

########## TASK 3 ##########
# Mainly Written by Anton, Aiga, Maria, Alberte, Geo

########## TASK 4 ##########
# Mainly Written by Mario, Alberte
"""

# GLOBAL CONTANTS ( global constants should always be written with caps. )
input_dim = (256, 256)
channel_dim = 3
CURRENT_FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = "group_8.pth"
EPOCHS = 10

# Define model
class group_8(nn.Module):
    def __init__(self):
        super(group_8, self).__init__()

        # Regarding the parameters we set the in_channels to 3 for best fit on RGB (Red,green blue) images
        # We are starting with a low degree of out_channels and increase this in our next layers
        # Our Kernel size is set to 3x3 for optimal fit that can capture the finer details in the 
        # The stride level is set to 1 which means the kernel moves with one pixel with each iterations, so with higher stride it would skip an iteration
        # We set padding to 1 to keep the size after the convulation. It secures that the spatial dimensatiosn remains consistent

        ## CONV LAYER
        self.conv1 = nn.Conv2d(in_channels=channel_dim, out_channels=16, kernel_size=3, stride=1, padding=1) # torch.Size([64, 16, 254, 254])

        #We are using maxpooling to reduce the spatial dimensions in order to reduce computational load, where we set our kernel size to 
        # 2x2 which should be appropriate for this type of images
        self.pool = nn.MaxPool2d(kernel_size=2) # torch.Size([64, 16, 127, 127])

        self.dropout = nn.Dropout(0.3)

        # The output size is 2 since we have two scenarios with disease or without disease
        # The hidden layer size we have set to respectively 512 followed by 256 which should be enough to capture the complex patterns

        self.fc1 = nn.Linear(in_features=16 * 128 * 128, out_features=512)  # Adjusted in_features and increased out_features
        self.fc2 = nn.Linear(in_features=512, out_features=256)            # Intermediate layer size
        self.fc3 = nn.Linear(in_features=256, out_features=2)                   


        # RELU is needed in CNNs in order to introducde non-linearity, which is needed for the netwrok to learn more complex patterns
        # The sigmoid function is used for binary classifications, which is strongly related to the logistic regression (again, for 2 outcomes)
        # Softmas is not really clear while we use this for binary classification, since its an activation for multi class problems

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


        # The forward method which  we are using here represents a common architecture that is suitable for classifications task
        # It takes in the parameter tunings which we have defined above and set to an appropriate amount related classifying pneumonia in lungs
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x) # After conv1 - x.shape : torch.Size([64, 16, 254, 254])
        # print(f"After conv1 - x.shape : {x.shape}")
        x = self.pool(x) # After maxpool - x.shape : torch.Size([64, 16, 127, 127])
        # print(f"After maxpool - x.shape : {x.shape}")
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))  ## RIDGE REGRESSION HERE
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x


# Function to move files to corresponding subfolder
def move_files(image_list, destination, source):
    for img_name in image_list:
        if 'normal' in img_name:
            category = 'normal'
        elif 'pneumonia' in img_name:
            category = 'pneumonia'
        else:
            print(f"Skipping {img_name}, as it doesn't match 'normal' or 'pneumonia'.")
            continue  # In case there's a file with an unexpected name format
        
        src = os.path.join(source, img_name)
        dst = os.path.join(destination, category, img_name)
        
        print(f"Moving {src} to {dst}")
        move(src, dst)

# Functions thats trains the model using our dataloader. 
def train(model, loader, criterion, optimizer):

    model.train()
    running_loss = 0.0

    # Training loop
    for batch, (images, labels) in enumerate(loader):
        print(f"Batch: {batch}")
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() # Clears any previous gradients.
        outputs = model(images)
        loss = criterion(outputs, labels)

        ## L2 Regularization (Ridge regression)
        l2_norm_fc2 = torch.norm(model.fc2.weight, p=2)  # Calculate L2 norm of the weights in the 2nd layer
        loss += 0.001 * l2_norm_fc2  # Add L2 regularization to the loss

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
        self.root = root # OK
        self.transform = transform # OK
        self.classes = ['normal', 'pneumonia'] # OK
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
    transforms.Resize(input_dim),  # Resize all images to 256x256 
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # For slightly altered medical images
    transforms.RandomHorizontalFlip(),  # Flip is useful since pneumonia might appear similary on either lung
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Adjust brightness and contrast slightly, since medical images ofent have it standardized.
    transforms.ToTensor(),
])
test_val_transform = transforms.Compose([
    transforms.Resize(input_dim),  # Resize all images to 256x256
    transforms.ToTensor(),
])


# Define paths
current_directory = os.path.dirname(os.path.abspath(__file__)) 
source_dir = current_directory + "/data" # Directory containing all the images ( make sure, that the data folder, is in the same folder as this file )
base_dir = current_directory + "/organized_data" # Base directory where organized images will be stored

# Create subdirectories for train, val, and test splits
for split in ['training', 'validation', 'testing']:
    for category in ['normal', 'pneumonia']:
        os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)

# Get list of all images
images = os.listdir(source_dir)

# Shuffle the images for a random split
random.shuffle(images)

# Calculate split sizes (e.g., 80% train, 10% val, 10% test)
total_images = len(images)
training_split = int(0.8 * total_images)
validation_split = int(0.9 * total_images)

# Move images to their respective directories
move_files(images[:training_split], os.path.join(base_dir, 'training'), source=source_dir)
move_files(images[training_split:validation_split], os.path.join(base_dir, 'validation'), source=source_dir)
move_files(images[validation_split:], os.path.join(base_dir, 'testing'), source=source_dir)

print("Images have been organized into subfolders successfully!")
#In order to see if the dataset is divided correctly we will visualize the distribution in a barchart.
# counting the files
def count_files(directory):
    return len(os.listdir(os.path.join(directory, 'normal'))), len(os.listdir(os.path.join(directory, 'pneumonia')))

train_counts = count_files(os.path.join(base_dir, 'training'))
val_counts = count_files(os.path.join(base_dir, 'validation'))
test_counts = count_files(os.path.join(base_dir, 'testing'))

# data for plotting
categories = ['Training', 'Validation', 'Testing']
normal_counts = [train_counts[0], val_counts[0], test_counts[0]]
pneumonia_counts = [train_counts[1], val_counts[1], test_counts[1]]

plt.figure(figsize=(10, 6))
y_pos = range(len(categories))

# horizontal bars
bars1 = plt.barh(y_pos, normal_counts, color='lightgreen', label='Normal')
bars2 = plt.barh(y_pos, pneumonia_counts, left=normal_counts, color='salmon', label='Pneumonia')

# annotations
for bar in bars1:
    plt.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
             str(int(bar.get_width())), ha='center', va='center', color='black')

for bar in bars2:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, 
             str(int(bar.get_width())), ha='center', va='center', color='black')

plt.xlabel('Number of Images')
plt.title('Distribution of Images in Folders')
plt.yticks(y_pos, categories)
plt.legend()
plt.show()


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
plt.savefig(f'{CURRENT_FILE_DIRECTORY}/group_8v2.png')
plt.show()

#### COMMENTS ON FINAL PICTURES ###
"""
We end up with an accuracy of around 94% on our validation data.
Overall, our model is learning quite well. But our learning rate may be too high, as we see large jumps
in the validation accuracy and loss.
Overall, the simplicity of our model, and the limited epochs that we run, we find that the model performs quite well.
Looking at the training loss, we generally see it decreaing and improving. The model fits the training data better and better.
However, we see fairly large jumps in the validation data.

"""

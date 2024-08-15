import os
from shutil import move
import random

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

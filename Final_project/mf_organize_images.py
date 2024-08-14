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
current_file_directory = os.path.dirname(os.path.abspath(__file__)) 
source_dir = current_file_directory + "/data" # Directory containing all the images ( make sure, that the data folder, is in the same folder as this file )
base_dir = current_file_directory + "/organized_data" # Base directory where organized images will be stored

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

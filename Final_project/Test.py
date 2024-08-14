""" 
Group 8 Memebers:
- Alberte Krogh Hansen
- Aiga Kalneja
- Anton Rindom
- Geo Vittrop Søndergaard
- Maria Post Hofmann
- Mario Hernández Festersen
"""

#                                   Task 1: 

# Imports
import os # Handling directory
from shutil import move # File operations
import random # Randomizing the data split


# mario har sat børnesikring 
# Directory defining
source_dir = r'/ADATA500/Programmering/Visual studio code/DS833 - DL/Uden navn/Final_project/data' 
# Create the base directory relative to the source directory
base_dir = os.path.join(os.path.dirname(source_dir), 'organized_data')

# Print the directories for verification
print(f"Source directory: {source_dir}")
print(f"Base directory: {base_dir}")

# Datastructure (Figure 1)
for split in ['training', 'validation', 'testing']:
    for category in ['normal', 'pneumonia']:
        os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)


# Creating and randonmizing an image list
images = os.listdir(source_dir)
random.shuffle(images)


# We chose to split into 80% training, 10% validation, 10% test, since 
# we want a large training set to reduze overfitting.
total_images = len(images)
training_split = int(0.8 * total_images) 
validation_split = int(0.9 * total_images) #culmulative number combined leaving 10% for testing

# Function to move files to corresponding subfolder
def move_files(image_list, dest_dir):
    for img_name in image_list:
        if 'normal' in img_name:
            category = 'normal'
        elif 'pneumonia' in img_name:
            category = 'pneumonia'
        else:
            print(f"Skipping {img_name}, as it doesn't match 'normal' or 'pneumonia'.")
            continue  # In case there's a file with an unexpected name format
        
        src = os.path.join(source_dir, img_name)
        dst = os.path.join(dest_dir, category, img_name)
        
        print(f"Moving {src} to {dst}")
        move(src, dst)

# Move images to their respective directories
move_files(images[:training_split], os.path.join(base_dir, 'training'))
move_files(images[training_split:validation_split], os.path.join(base_dir, 'validation'))
move_files(images[validation_split:], os.path.join(base_dir, 'testing'))

print("Images have been organized into subfolders successfully!")

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

# Directory defining
source_dir = os.getcwd() + '/Final_project/data' 
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

move_files(images[:training_split], os.path.join(base_dir, 'training'))
move_files(images[training_split:validation_split], os.path.join(base_dir, 'validation'))
move_files(images[validation_split:], os.path.join(base_dir, 'testing'))

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

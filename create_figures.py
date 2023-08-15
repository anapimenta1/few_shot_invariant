from PIL import Image
import os
import random
import shutil

# Replace with the list of directories containing your class subdirectories
base_dirs = ['/home/guests2/apa/datasets/crc-tp/Fold1/Training',
    '/home/guests2/apa/datasets/nct/NCT-CRC-HE-100K',
    '/home/guests2/apa/datasets/lc25000/all',
    '/home/guests2/apa/datasets/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/all']

# Define the path to the output directory
output_dir = '/home/guests2/apa/few_shot/figures'

# Process each base directory
for base_dir in base_dirs:
    # Get the name of the dataset (name of the folder after '/home/guests2/apa/datasets')
    dataset_name = base_dir.split('/')[5]

    # Create a directory for the dataset inside the output directory
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Get a list of class subdirectories
    class_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Save one image from each class in the dataset's output directory
    for class_dir in class_dirs:
        class_path = os.path.join(base_dir, class_dir)
        class_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

        if class_images:
            random_image = random.choice(class_images)
            image_path = os.path.join(class_path, random_image)
            output_image_path = os.path.join(dataset_output_dir, f'{class_dir}.png')
            shutil.copy(image_path, output_image_path)

    print(f"Images from '{dataset_name}' dataset saved in '{dataset_output_dir}'")
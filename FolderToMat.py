from PIL import Image
import numpy as np
from scipy.io import savemat
import os

def yolo_to_mat(yolo_file_path, image_path, output_mat_path):
    # Read the image to get its size
    with Image.open(image_path) as img:
        image_size = img.size  # (width, height)

    # Read YOLO annotations
    with open(yolo_file_path, 'r') as file:
        yolo_data = file.readlines()

    # Convert YOLO data to MAT format
    locations = []
    for line in yolo_data:
        _, x_center, y_center, _, _ = map(float, line.split())
        abs_x_center = x_center * image_size[0]
        abs_y_center = y_center * image_size[1]
        locations.append([abs_x_center, abs_y_center])

    # Convert list to numpy array and structure for saving
    locations_array = np.array(locations, dtype=np.float32).reshape(-1, 2)
    number_array = np.array([len(locations)], dtype=np.uint8)
    struct_array = np.array([[(locations_array, number_array)]], dtype=[('location', 'O'), ('number', 'O')])
    image_info = np.empty((1, 1), dtype=object)
    image_info[0, 0] = struct_array
    mat_data = {'image_info': image_info}

    # Save to MAT file
    savemat(output_mat_path, mat_data)

def process_folder(folder_path):
    # List all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            base_name = os.path.splitext(file_name)[0]
            image_path = os.path.join(folder_path, file_name)
            yolo_file_path = os.path.join(folder_path, base_name + '.txt')
            output_mat_path = os.path.join(folder_path, base_name + '_ann.mat')

            if os.path.exists(yolo_file_path):
                yolo_to_mat(yolo_file_path, image_path, output_mat_path)
            else:
                print(f"Warning: Annotation file does not exist for {image_path}")

# Example usage
folder_path = 'D:\\Github\\YoloToMat\\ann'  # The path to the folder containing images and annotations
process_folder(folder_path)

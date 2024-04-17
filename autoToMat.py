from PIL import Image
import numpy as np
from scipy.io import savemat
import os

def yolo_to_mat(yolo_file_path, image_path, output_mat_path):
    with Image.open(image_path) as img:
        image_size = img.size  # (width, height)
    with open(yolo_file_path, 'r') as file:
        yolo_data = file.readlines()
    locations = []
    for line in yolo_data:
        _, x_center, y_center, width, height = map(float, line.split())
        abs_x_center = x_center * image_size[0]
        abs_y_center = y_center * image_size[1]
        abs_width = width * image_size[0]
        abs_height = height * image_size[1]
        locations.append([abs_x_center - abs_width / 2, abs_y_center - abs_height / 2,
                          abs_x_center + abs_width / 2, abs_y_center + abs_height / 2])
    mat_data = {
        'image_info': np.array([[(np.array(locations), np.array([[len(locations)]]))]],
                               dtype=[('location', 'O'), ('number', 'O')])
    }
    savemat(output_mat_path, mat_data)

def process_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(folder_path, file)
            yolo_file_path = os.path.splitext(image_path)[0] + '.txt'
            if os.path.exists(yolo_file_path):
                output_mat_path = os.path.splitext(image_path)[0] + '.mat'
                yolo_to_mat(yolo_file_path, image_path, output_mat_path)
                print(f"Converted: {file}")

# Specify your folder path here
folder_path = 'path/to/your/images_and_annotations'
process_folder(folder_path)

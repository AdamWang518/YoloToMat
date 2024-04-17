from PIL import Image
import numpy as np
from scipy.io import savemat

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
        # Convert relative to absolute coordinates
        abs_x_center = x_center * image_size[0]
        abs_y_center = y_center * image_size[1]
        # Only store the center coordinates
        locations.append([abs_x_center, abs_y_center])

    # Ensure correct data structuring for MAT file
    locations_array = np.array(locations, dtype=np.float32).reshape(-1, 2)
    number_array = np.array([len(locations)], dtype=np.uint8)

    # Create the nested structure with the three layers and correct dtype
    struct_array = np.array([[(locations_array, number_array)]],
                            dtype=[('location', 'O'), ('number', 'O')])
    image_info = np.empty((1, 1), dtype=object)
    image_info[0, 0] = struct_array

    # Save to MAT file with proper structure
    mat_data = {'image_info': image_info}
    savemat(output_mat_path, mat_data)

# Example usage
yolo_file_path = 'D:\\Github\\YoloToMat\\ann\\153.txt'
image_path = 'D:\\Github\\YoloToMat\\ann\\153.png'  # The path to the image file
output_mat_path = 'D:\\Github\\YoloToMat\\ann\\153_ann.mat'

yolo_to_mat(yolo_file_path, image_path, output_mat_path)

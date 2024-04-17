from PIL import Image
import numpy as np
from scipy.io import savemat

def yolo_to_mat(yolo_file_path, image_path, output_mat_path):
    """
    Convert YOLO annotation format to the specified MAT format, with automatic image size detection.
    
    Parameters:
    - yolo_file_path: Path to the YOLO format file.
    - image_path: Path to the image file (to read its size).
    - output_mat_path: Path to save the output MAT file.
    """
    # Read the image to get its size
    with Image.open(image_path) as img:
        image_size = img.size  # (width, height)

    # Read YOLO annotations
    with open(yolo_file_path, 'r') as file:
        yolo_data = file.readlines()

    # Convert YOLO data to MAT format
    locations = []
    for line in yolo_data:
        _, x_center, y_center, width, height = map(float, line.split())
        # Convert relative to absolute coordinates
        abs_x_center = x_center * image_size[0]
        abs_y_center = y_center * image_size[1]
        abs_width = width * image_size[0]
        abs_height = height * image_size[1]
        # Assuming the location is based on the box's center and dimensions
        locations.append([abs_x_center - abs_width / 2, abs_y_center - abs_height / 2,
                          abs_x_center + abs_width / 2, abs_y_center + abs_height / 2])

    # Prepare MAT file structure
    mat_data = {
        'image_info': np.array([[(np.array(locations), np.array([[len(locations)]]))]],
                               dtype=[('location', 'O'), ('number', 'O')])
    }

    # Save to MAT file
    savemat(output_mat_path, mat_data)

# Example usage
yolo_file_path = 'path/to/yolo_annotation.txt'
image_path = 'path/to/image_file.jpg'  # The path to the image file
output_mat_path = 'path/to/output_annotation.mat'

yolo_to_mat(yolo_file_path, image_path, output_mat_path)

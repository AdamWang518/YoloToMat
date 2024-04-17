import scipy.io as sio

def inspect_mat_file(file_path):
    """
    Load and inspect the content of a .mat file.
    
    Parameters:
    - file_path: Path to the .mat file.
    
    Returns:
    - A dictionary with keys and details about each variable stored in the .mat file.
    """
    data = sio.loadmat(file_path)  # Load the MAT file
    data_info = {}
    
    for key, value in data.items():
        if key not in ['__header__', '__version__', '__globals__']:  # Skip meta data
            # Get more details about the variable
            data_info[key] = {
                'type': type(value),
                'shape': value.shape if hasattr(value, 'shape') else None,
                'dtype': value.dtype if hasattr(value, 'dtype') else None,
                'content': value if hasattr(value, 'shape') and value.size < 10 else 'Data too large to display'
            }
    
    return data_info

# Usage example
file_path = 'D:\\Github\\YoloToMat\\153_ann.mat'
inspect_result = inspect_mat_file(file_path)
print(inspect_result)

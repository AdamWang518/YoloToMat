from PIL import Image
import numpy as np
from scipy.io import savemat
import os
import shutil  # 導入shutil模塊用於文件複製

def yolo_to_mat(yolo_file_path, image_path, output_mat_path):
    # 讀取圖片以獲得其尺寸
    with Image.open(image_path) as img:
        image_size = img.size  # (width, height)

    # 讀取YOLO標註
    with open(yolo_file_path, 'r') as file:
        yolo_data = file.readlines()

    # 將YOLO數據轉換為MAT格式
    locations = []
    for line in yolo_data:
        _, x_center, y_center, _, _ = map(float, line.split())
        abs_x_center = x_center * image_size[0]
        abs_y_center = y_center * image_size[1]
        locations.append([abs_x_center, abs_y_center])

    # 將列表轉換為numpy數組並結構化保存
    locations_array = np.array(locations, dtype=np.float32).reshape(-1, 2)
    number_array = np.array([len(locations)], dtype=np.uint8)
    struct_array = np.array([[(locations_array, number_array)]], dtype=[('location', 'O'), ('number', 'O')])
    image_info = np.empty((1, 1), dtype=object)
    image_info[0, 0] = struct_array
    mat_data = {'image_info': image_info}

    # 保存到MAT檔案
    savemat(output_mat_path, mat_data)

def process_folder(folder_path, output_folder):
    # 確保輸出資料夾存在，如果不存在，則創建它
    os.makedirs(output_folder, exist_ok=True)

    # 列出資料夾中的所有檔案
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            base_name = os.path.splitext(file_name)[0]
            image_path = os.path.join(folder_path, file_name)
            yolo_file_path = os.path.join(folder_path, base_name + '.txt')
            output_mat_path = os.path.join(output_folder, 'GT_{}.mat'.format(base_name))
            output_image_path = os.path.join(output_folder, file_name)

            if os.path.exists(yolo_file_path):
                yolo_to_mat(yolo_file_path, image_path, output_mat_path)
                shutil.copy(image_path, output_image_path)  # 複製圖片文件到輸出資料夾
            else:
                print(f"警告：{image_path}缺少標註文件")


# 示例用法
folder_path = 'C:\\Users\\User\\Pictures\\RandomLabel'  # 包含圖片和標註的文件夾路徑
output_folder = 'C:\\Users\\User\\Pictures\\UnderWater'  # 指定輸出MAT檔案的目錄
process_folder(folder_path, output_folder)

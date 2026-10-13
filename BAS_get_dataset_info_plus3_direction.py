import pickle
import re
import numpy as np

# 读取 info_BAS.txt 文件
def parse_info_file(file_path):
    image_info = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"Image Path: .*/(\d+).nii.gz, Image Size: \((\d+), (\d+), (\d+)\)", line)
            if match:
                img_id = match.group(1)
                x_size, y_size, z_size = int(match.group(2)), int(match.group(3)), int(match.group(4))
                image_info[img_id] = (x_size, y_size, z_size)
    return image_info

# 计算每个切块在原始图像中的位置
def calculate_block_positions(image_size, crop_cube_size, stride):
    x_size, y_size, z_size = image_size
    crop_x, crop_y, crop_z = crop_cube_size
    stride_x, stride_y, stride_z = stride

    positions = []
    count = 0
    for z in range(0, z_size, stride_z):
        for y in range(0, y_size, stride_y):
            for x in range(0, x_size, stride_x):
                z_start = min(z, z_size - crop_z)
                y_start = min(y, y_size - crop_y)
                x_start = min(x, x_size - crop_x)
                positions.append((count, (x_start, y_start, z_start)))
                count += 1
    return positions

# 加载 pkl 数据
def load_pkl_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# 保存 pkl 数据
def save_pkl_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# 主程序
info_file_path = 'dataset_info/info_BAS.txt'
pkl_file_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128.pkl'
output_pkl_file_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_positions_and_ids.pkl'

image_info = parse_info_file(info_file_path)
data_dict = load_pkl_data(pkl_file_path)

crop_cube_size = (128, 128, 128)  # 块大小
stride = (64, 64, 64)  # 步长

# 遍历每个切块并添加位置信息、图像ID和块编号
for key in data_dict.keys():
    img_id_match = re.match(r"BAS_(\d+)_\d+", key)  # 提取图像 ID

    # 结构强度
    data_dict[key]["boundary_inner_ratio"] = data_dict[key]["airway_pixel_num_boundary"] / data_dict[key]["airway_pixel_num_inner"] if data_dict[key]["airway_pixel_num_inner"] != 0 else np.nan

        # data_dict[key]["boundary_inner_ratio"] = np.nan
    # 体素强度
    data_dict[key]["boundary_inner_contrast"] = data_dict[key]["boundary_strength_mean"] - \
                                                               data_dict[key]["inner_strength_mean"]
    data_dict[key]["airway_non_airway_mean_deta"] = data_dict[key][
                                                                       "airway_strength_mean"] - \
                                                                   data_dict[key][
                                                                       "non_airway_strength_mean"]

    if img_id_match:
        img_id = img_id_match.group(1)
        if img_id in image_info:
            image_size = image_info[img_id]
            block_id = int(key.split('_')[-1])  # 从名称中提取块编号
            positions = calculate_block_positions(image_size, crop_cube_size, stride)

            # 查找并添加位置
            for pos in positions:
                if pos[0] == block_id:
                    # 添加位置信息、图像ID和块编号
                    data_dict[key]["position"] = pos[1]
                    data_dict[key]["image_id"] = img_id
                    data_dict[key]["block_id"] = block_id
                    data_dict[key]["image_size"] = image_size
                    print(f"Added position, image_id, and block_id for {key}: {pos[1]}, {img_id}, {block_id}, {image_size}")
                    break
            else:
                print(f"Warning: No position found for block {key}")  # 若未找到位置
        else:
            print(f"Warning: Image ID {img_id} not found in info_BAS.txt")
    else:
        print(f"Warning: Could not extract image ID from {key}")

# 保存更新后的数据
save_pkl_data(data_dict, output_pkl_file_path)

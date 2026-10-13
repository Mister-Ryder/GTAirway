import os
import numpy as np
import skimage.io as io
import SimpleITK as sitk
import tqdm

import SimpleITK as sitk
import numpy as np
import os

'''
def normalize_image(image):
    """标准化图像：将值范围归一化到[0, 1]区间"""
    array = sitk.GetArrayFromImage(image)
    min_val = np.min(array)
    max_val = np.max(array)
    # 防止分母为0的情况
    if max_val - min_val == 0:
        return image
    normalized_array = (array - min_val) / (max_val - min_val)
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)
    return normalized_image

def preprocess_image(image):
    """对图像进行预处理：用最大值减去每个点，并将减去后是最大值的点设置为0"""
    array = sitk.GetArrayFromImage(image)
    max_val = np.max(array)
    # 用最大值减去数组中的每个值
    preprocessed_array = max_val - array
    # 将现在最大值的地方设置为0
    new_max_val = np.max(preprocessed_array)
    preprocessed_array[preprocessed_array == new_max_val] = 0
    preprocessed_image = sitk.GetImageFromArray(preprocessed_array)
    preprocessed_image.CopyInformation(image)
    return preprocessed_image

def process_images(directory, output_directory):
    """处理目录中的所有图像并逐个保存，包含标准化和预处理功能"""
    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]
    for file_name in files:
        image = sitk.ReadImage(os.path.join(directory, file_name))
        normalized_image = normalize_image(image)
        preprocessed_image = preprocess_image(normalized_image)
        sitk.WriteImage(preprocessed_image, os.path.join(output_directory, file_name))

# 使用函数
input_directory = '/home/ubuntu/shared-data/BAS/train/train_map_gaussian3/'
output_directory = '/home/ubuntu/shared-data/BAS/train/train_map_gaussian3_fan/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

process_images(input_directory, output_directory)

print("fan finish")


import SimpleITK as sitk
import numpy as np
import os

def normalize_image(image):
    """标准化图像：将值范围归一化到[0, 1]区间"""
    array = sitk.GetArrayFromImage(image)
    min_val = np.min(array)
    max_val = np.max(array)
    # 防止分母为0的情况
    if max_val - min_val == 0:
        return image
    normalized_array = (array - min_val) / (max_val - min_val)
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)
    return normalized_image

def process_images(directory, output_directory):
    """处理目录中的所有图像并逐个保存，包含标准化和预处理功能"""
    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]
    for file_name in files:
        image = sitk.ReadImage(os.path.join(directory, file_name))
        normalized_image = normalize_image(image)
        # preprocessed_image = preprocess_image(normalized_image)
        sitk.WriteImage(normalized_image , os.path.join(output_directory, file_name))

# 使用函数
input_directory = '/home/ubuntu/shared-data/BAS/train/train_map_gaussian3'
output_directory = '/home/ubuntu/shared-data/BAS/train/train_map_gaussian3_zheng/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

process_images(input_directory, output_directory)

print("zheng finish")
'''
# build the raw_data_dict for train

raw_data_dict = dict()

# # BAS TRAIN DATA

raw_data_dict = dict()

BAS_skeleton_file_path = "/home/ubuntu/shared-data/BAS/train"
BAS_train_skeleton_path = BAS_skeleton_file_path + "/train_label_skeleton3"
BAS_train_distance_path = BAS_skeleton_file_path + "/train_map_distance3"
BAS_train_gaussian_path = BAS_skeleton_file_path + "/train_map_gaussian3_fan"
BAS_train_gaussian_path2 = BAS_skeleton_file_path + "/train_map_gaussian3_zheng"

BAS_skeleton_names = os.listdir(BAS_train_skeleton_path)
BAS_skeleton_names.sort()
#
BAS_distance_names = os.listdir(BAS_train_distance_path)
BAS_distance_names.sort()

BAS_gaussian_names = os.listdir(BAS_train_gaussian_path)
BAS_gaussian_names.sort()

BAS_gaussian_names2 = os.listdir(BAS_train_gaussian_path2)
BAS_gaussian_names2.sort()

case_names = []

for case in BAS_skeleton_names:
    temp = case.split(".")[0]
    case_names.append(temp)
    raw_data_dict["BAS_" + temp] = {}
    raw_data_dict["BAS_" + temp]["train_label_skeleton"] = BAS_train_skeleton_path + "/" + case

for case in BAS_distance_names:
    temp = case.split(".")[0]
    if temp in case_names:
        raw_data_dict["BAS_" + temp]["train_map_distance"] = BAS_train_distance_path + "/" + case

for case in BAS_gaussian_names:
    temp = case.split(".")[0]
    if temp in case_names:
        raw_data_dict["BAS_" + temp]["train_map_gaussian"] = BAS_train_gaussian_path + "/" + case

for case in BAS_gaussian_names2:
    temp = case.split(".")[0]
    if temp in case_names:
        raw_data_dict["BAS_" + temp]["train_map_gaussian2"] = BAS_train_gaussian_path2 + "/" + case

BAS_skeleton_data_dict = raw_data_dict

print(BAS_skeleton_data_dict)


def crop_one_3d_img(input_img, crop_cube_size, stride):
    # input_img: 3d matrix, numpy.array
    assert isinstance(crop_cube_size, (int, tuple))
    if isinstance(crop_cube_size, int):
        crop_cube_size = np.array([crop_cube_size, crop_cube_size, crop_cube_size])
    else:
        assert len(crop_cube_size) == 3

    crop_cube_size = (min(crop_cube_size[0], input_img.shape[0]),
                      min(crop_cube_size[1], input_img.shape[1]),
                      min(crop_cube_size[2], input_img.shape[2]))

    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
        stride = np.array([stride, stride, stride])
    else:
        assert len(stride) == 3

    img_shape = input_img.shape

    total = len(np.arange(0, img_shape[0], stride[0])) * len(np.arange(0, img_shape[1], stride[1])) * len(
        np.arange(0, img_shape[2], stride[2]))

    count = 0

    crop_list = []

    for i in np.arange(0, img_shape[0], stride[0]):
        for j in np.arange(0, img_shape[1], stride[1]):
            for k in np.arange(0, img_shape[2], stride[2]):
                print('crop one 3d img progress : ' + str(np.int(count / total * 100)) + '%', end='\r')
                if i + crop_cube_size[0] <= img_shape[0]:
                    x_start_input = i
                    x_end_input = i + crop_cube_size[0]
                    x_start_output = i
                    x_end_output = i + stride[0]
                else:
                    x_start_input = img_shape[0] - crop_cube_size[0]
                    x_end_input = img_shape[0]
                    x_start_output = i
                    x_end_output = img_shape[0]

                if j + crop_cube_size[1] <= img_shape[1]:
                    y_start_input = j
                    y_end_input = j + crop_cube_size[1]
                    y_start_output = j
                    y_end_output = j + stride[1]
                else:
                    y_start_input = img_shape[1] - crop_cube_size[1]
                    y_end_input = img_shape[1]
                    y_start_output = j
                    y_end_output = img_shape[1]

                if k + crop_cube_size[2] <= img_shape[2]:
                    z_start_input = k
                    z_end_input = k + crop_cube_size[2]
                    z_start_output = k
                    z_end_output = k + stride[2]
                else:
                    z_start_input = img_shape[2] - crop_cube_size[2]
                    z_end_input = img_shape[2]
                    z_start_output = k
                    z_end_output = img_shape[2]

                crop_temp = input_img[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]
                crop_list.append(np.array(crop_temp, dtype=np.float))

                count = count + 1

    return crop_list

#### pre crop EXACT09

crop_cube_size = (128, 128, 128)
stride = (64, 64, 64)

# -----INPUT-----
output_file_path = "/home/ubuntu/shared-data/Precrop_dataset/Precrop_dataset_for_BAS"

if not os.path.exists(output_file_path + "/skeleton/"):
    os.makedirs(output_file_path + "/skeleton/")

if not os.path.exists(output_file_path + "/distance/"):
    os.makedirs(output_file_path + "/distance/")

if not os.path.exists(output_file_path + "/gaussian/"):
    os.makedirs(output_file_path + "/gaussian/")

if not os.path.exists(output_file_path + "/gaussian2/"):
    os.makedirs(output_file_path + "/gaussian2/")

raw_data_dict = BAS_skeleton_data_dict
# -----END-----

for i, case in enumerate(raw_data_dict.keys()):
    print(case)
    skeleton_img = io.imread(raw_data_dict[case]["train_label_skeleton"], plugin='simpleitk')
    distance_img = io.imread(raw_data_dict[case]["train_map_distance"], plugin='simpleitk')
    gaussian_img = io.imread(raw_data_dict[case]["train_map_gaussian"], plugin='simpleitk')
    gaussian_img2 = io.imread(raw_data_dict[case]["train_map_gaussian2"], plugin='simpleitk')

    skeleton_img_crop_list = crop_one_3d_img(skeleton_img, crop_cube_size=crop_cube_size, stride=stride)
    distance_img_crop_list = crop_one_3d_img(distance_img, crop_cube_size=crop_cube_size, stride=stride)
    gaussian_img_crop_list = crop_one_3d_img(gaussian_img, crop_cube_size=crop_cube_size, stride=stride)
    gaussian_img_crop_list2 = crop_one_3d_img(gaussian_img2, crop_cube_size=crop_cube_size, stride=stride)

    # assert len(skeleton_img_crop_list) == len(distance_img_crop_list)

    for idx in range(len(gaussian_img_crop_list)):
        print("progress: " + str(idx) + "th crop | " + str(i) + "th 3d img: " + str(case), end="\r")

        # sitk.WriteImage(sitk.GetImageFromArray(raw_img_crop_list[idx]), output_file_path+"/image/"+case+"_"+str(idx)+".nii.gz")
        # sitk.WriteImage(sitk.GetImageFromArray(label_img_crop_list[idx]), output_file_path+"/label/"+case+"_"+str(idx)+".nii.gz")

        np.save(output_file_path + "/skeleton/" + case + "_" + str(idx) + ".npy", skeleton_img_crop_list[idx])
        np.save(output_file_path + "/distance/" + case + "_" + str(idx) + ".npy", distance_img_crop_list[idx])
        np.save(output_file_path + "/gaussian/" + case + "_" + str(idx) + ".npy", gaussian_img_crop_list[idx])#FAN
        np.save(output_file_path + "/gaussian2/" + case + "_" + str(idx) + ".npy", gaussian_img_crop_list2[idx])#ZHENG


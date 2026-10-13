import os
import numpy as np
import skimage.io as io
import SimpleITK as sitk

# build the raw_data_dict for train

raw_data_dict = dict()

# EXACT09 train data

raw_data_dict = dict()

BAS_file_path = "/home/ubuntu/shared-data/BAS/train"
BAS_train_raw_path = BAS_file_path + "/image"
BAS_train_label_path = BAS_file_path + "/label"

BAS_raw_names = os.listdir(BAS_train_raw_path)
BAS_raw_names.sort()

BAS_label_names = os.listdir(BAS_train_label_path)
BAS_label_names.sort()

case_names = []

for case in BAS_raw_names:
    temp = case.split(".")[0]
    case_names.append(temp)
    raw_data_dict["BAS_" + temp] = {}
    raw_data_dict["BAS_" + temp]["image"] = BAS_train_raw_path + "/" + case

for case in BAS_label_names:
    temp = case.split(".")[0]
    if temp in case_names:
        raw_data_dict["BAS_" + temp]["label"] = BAS_train_label_path + "/" + case

BAS_data_dict = raw_data_dict

# print(LIDC_IDRI_data_dict)
print(BAS_data_dict)

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

crop_cube_size = (128, 128, 128)
stride = (64, 64, 64)

# -----INPUT-----
output_file_path = "/home/ubuntu/shared-data/Precrop_dataset/Precrop_dataset_for_BAS"

if not os.path.exists(output_file_path + "/image/"):
    os.makedirs(output_file_path + "/image/")

if not os.path.exists(output_file_path + "/label/"):
    os.makedirs(output_file_path + "/label/")

raw_data_dict = BAS_data_dict
# -----END-----

for i, case in enumerate(raw_data_dict.keys()):
    # raw_img = io.imread(raw_data_dict[case]["image"], plugin='simpleitk')
    label_img = io.imread(raw_data_dict[case]["label"], plugin='simpleitk')
    #使用这个方式加载niigz格式数据得到普通的 3D 图像数组，方向、原点信息被默认清除。

    # raw_img_crop_list = crop_one_3d_img(raw_img, crop_cube_size=crop_cube_size, stride=stride)
    label_img_crop_list = crop_one_3d_img(label_img, crop_cube_size=crop_cube_size, stride=stride)

    # assert len(raw_img_crop_list) == len(label_img_crop_list)

    for idx in range(len(label_img_crop_list)):
        print("progress: " + str(idx) + "th crop | " + str(i) + "th 3d img: " + str(case), end="\r")

        # sitk.WriteImage(sitk.GetImageFromArray(raw_img_crop_list[idx]), output_file_path+"/image/"+case+"_"+str(idx)+".nii.gz")
        # sitk.WriteImage(sitk.GetImageFromArray(label_img_crop_list[idx]), output_file_path+"/label/"+case+"_"+str(idx)+".nii.gz")

        # np.save(output_file_path + "/image/" + case + "_" + str(idx) + ".npy", raw_img_crop_list[idx])
        np.save(output_file_path + "/label/" + case + "_" + str(idx) + ".npy", label_img_crop_list[idx])
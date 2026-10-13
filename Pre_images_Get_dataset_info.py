# -*- coding: utf-8 -*-
# @Time    : 2024/3/17 21:57
# @Author  : JIA
# @FileName: Pre_images_Get_dataset_info.py
# @Software: PyCharm
# @Blog    ：

import os
import numpy as np
import skimage.io as io
import SimpleITK as sitk

import pickle
import edt  # cython package
import tqdm


# build the raw_data_dict for train
# EXACT09 train data

raw_data_dict = dict()

EXACT09_file_path = "/mnt/e/00-Data/Airway/EXACT09/TEMP"#"/data/Airway/EXACT09_3D"
EXACT09_train_raw_path = EXACT09_file_path+"/train"
EXACT09_train_label_path = EXACT09_file_path+"/train_label"

EXACT09_raw_names = os.listdir(EXACT09_train_raw_path)
EXACT09_raw_names.sort()

EXACT09_label_names = os.listdir(EXACT09_train_label_path)
EXACT09_label_names.sort()

case_names = []

for case in EXACT09_raw_names:
    temp = case.split(".")[0]
    case_names.append(temp)
    raw_data_dict["EXACT09_"+temp]={}
    raw_data_dict["EXACT09_"+temp]["image"]=EXACT09_train_raw_path+"/"+case

for case in EXACT09_label_names:
    temp = case.split("_")[0]
    if temp in case_names:
        raw_data_dict["EXACT09_"+temp]["label"]=EXACT09_train_label_path+"/"+case


output_file_path = "Precrop_dataset_for_EXACT09_no_crop"

if not os.path.exists(output_file_path + "/image/"):
    os.makedirs(output_file_path + "/image/")

if not os.path.exists(output_file_path + "/label/"):
    os.makedirs(output_file_path + "/label/")

# -----END-----
#使用tqdm加一个进度条
for i,case in tqdm.tqdm(enumerate(raw_data_dict.keys()), total=len(raw_data_dict.keys())):

    raw_img = io.imread(raw_data_dict[case]["image"], plugin='simpleitk')
    label_img = io.imread(raw_data_dict[case]["label"], plugin='simpleitk')
    np.save(output_file_path + "/image/" + case + ".npy", raw_img)
    np.save(output_file_path + "/label/" + case + ".npy", label_img)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# ### Get pre-cropped EXACT09 dataset info

Precrop_dataset_for_train_path = output_file_path #"./Precrop_dataset_for_EXACT09"
Precrop_dataset_for_train_raw_path = Precrop_dataset_for_train_path + "/image"
Precrop_dataset_for_train_label_path = Precrop_dataset_for_train_path + "/label"

raw_case_name_list = os.listdir(Precrop_dataset_for_train_raw_path)
label_case_name_list = os.listdir(Precrop_dataset_for_train_label_path)

assert raw_case_name_list == label_case_name_list

data_dict = dict()

# for idx, name in enumerate(raw_case_name_list):
for idx, name in tqdm.tqdm(enumerate(raw_case_name_list), total=len(raw_case_name_list)):
    print(f"\rProcessing {name} ({idx + 1}/{len(raw_case_name_list)})", end="")
    # print("process "+str(name)+" | "+str(idx/len(raw_case_name_list)), end="\r")
    data_dict[name.split(".")[0]] = {}
    data_dict[name.split(".")[0]]["image"] = Precrop_dataset_for_train_raw_path + "/" + name
    data_dict[name.split(".")[0]]["label"] = Precrop_dataset_for_train_label_path + "/" + name
    label_temp = np.load(Precrop_dataset_for_train_label_path + "/" + name)
    data_dict[name.split(".")[0]]["airway_pixel_num"] = np.sum(label_temp)

    label_temp = np.array(label_temp, dtype=np.uint32, order='F')
    # 通过edt计算label_temp的距离变换，得到边界和内部的像素数
    label_temp_edt = edt.edt(
        label_temp,
        black_border=True, order='F',
        parallel=1)

    data_dict[name.split(".")[0]]["airway_pixel_num_boundary"] = len(np.where(label_temp_edt == 1)[0])
    data_dict[name.split(".")[0]]["airway_pixel_num_inner"] = len(np.where(label_temp_edt > 1)[0])

save_obj(data_dict, "./dataset_info/dataset_info_EXACT09_no_crop")

#################################  test数据集制作  ###############################################
# ### Get the case names of EXACT09

EXACT_img_niigz_path = "/mnt/e/00-Data/Airway/EXACT09/TEMP/train"
EXACT_label_niigz_path = "/mnt/e/00-Data/Airway/EXACT09/TEMP/train_label"

EXACT_names = os.listdir(EXACT_img_niigz_path)
EXACT_names.sort()
EXACT_label_names = os.listdir(EXACT_label_niigz_path)
EXACT_label_names.sort()
print(EXACT_names, EXACT_label_names)

EXACT09_names = []
for EXACT09_name in EXACT_names:
    EXACT09_names.append("EXACT09_" + EXACT09_name.split(".")[0])
EXACT09_names = np.array(EXACT09_names)
EXACT09_names = np.unique(EXACT09_names)

# ### Split train/test set
# names = np.concatenate((EXACT09_names, LIDC_IDRI_names))# concatenate the names of EXACT09 and LIDC-IDRI 把EXACT09和LIDC-IDRI的名字连接起来
# you can split train/test by yourself
# just show an example
names = EXACT09_names
# test_names = ['LIDC_IDRI_0698', 'LIDC_IDRI_0710', 'LIDC_IDRI_0810',
#         'LIDC_IDRI_0376', 'EXACT09_CASE13', 'LIDC_IDRI_1004',
#         'EXACT09_CASE08', 'EXACT09_CASE01', 'EXACT09_CASE05',
#         'LIDC_IDRI_0744']
test_names = ['EXACT09_CASE13', 'LIDC_IDRI_1004',
              'EXACT09_CASE08', 'EXACT09_CASE01', 'EXACT09_CASE05']

#print("test name: " + test_names)
train_names = []
for name in names:
    if name not in test_names:
        train_names.append(name)
train_names = np.array(train_names)
#print("train names: " + train_names)

# ### Get dataset dict for training

data_dict_EXACT09_128 = data_dict #load_obj("dataset_info/dataset_info_EXACT09_crops_128")

train_test_set_dict_EXACT09_LIDC_IDRI_128 = {}
train_test_set_dict_EXACT09_LIDC_IDRI_128["train"] = {}
train_test_set_dict_EXACT09_LIDC_IDRI_128["test"] = {}

for case in data_dict_EXACT09_128.keys():
    if (case.split("_")[0] + "_" + case.split("_")[1]) in train_names:
        train_test_set_dict_EXACT09_LIDC_IDRI_128["train"][case] = data_dict_EXACT09_128[case]
    elif (case.split("_")[0] + "_" + case.split("_")[1]) in test_names:
        train_test_set_dict_EXACT09_LIDC_IDRI_128["test"][case] = data_dict_EXACT09_128[case]

train_test_set_dict_EXACT09_LIDC_IDRI_128["train_names"] = train_names
train_test_set_dict_EXACT09_LIDC_IDRI_128["test_names"] = test_names


########################## Get dataset_info  ##########################
import copy
data_dict_org = train_test_set_dict_EXACT09_LIDC_IDRI_128['train']
data_dict_extended = copy.deepcopy(data_dict_org) # copy.deepcopy()是深拷贝，即新建一个一模一样的对象，但是不是同一个对象
save_obj(data_dict_extended,"dataset_info/train_dataset_info_EXACT09_LIDC_IDRI_no_crop")


"""
########################## Get dataset_info for iterative training strategy.  ##########################
# First, train with higher frequency on airways of high generations.
# Next, train with higher frequency on airways of low generations. And repeat the process.

data_dict_org = train_test_set_dict_EXACT09_LIDC_IDRI_128['train']

import copy

data_dict_extended = copy.deepcopy(data_dict_org)

is_more_big = True  # higher freq on airways of low gen (thicker airways)
copy_times_I = 10

# 我想把enmrated改为tqdm，显示进度条
# for idx, case in enumerate(data_dict_org.keys()):
## 通过对每个数据点根据其内部和边界呼吸道像素数量的比例，以及可能的其他外部条件（如is_more_big和copy_times_I的值），
# 来决定复制的次数，并将这些复制项扩展到新的数据字典中，从而可能是为了数据增强或平衡数据集。
for idx, case in tqdm.tqdm(enumerate(data_dict_org.keys()), total=len(data_dict_org.keys())):
    print(f"\rProcessing {name} ({idx + 1}/{len(data_dict_org.keys())})", end="")
    if data_dict_org[case]["airway_pixel_num"] > 0:
        if is_more_big:
            copy_times_II = np.ceil(
                data_dict_org[case]["airway_pixel_num_inner"] / data_dict_org[case]["airway_pixel_num_boundary"])
        else:
            if data_dict_org[case]["airway_pixel_num_inner"] == 0:
                copy_times_II = np.ceil(data_dict_org[case]["airway_pixel_num_boundary"])
            else:
                copy_times_II = np.ceil(
                    data_dict_org[case]["airway_pixel_num_boundary"] / data_dict_org[case]["airway_pixel_num_inner"])

        for i in range(int(copy_times_I * copy_times_II)):
            data_dict_extended[case + "_copy_" + str(i + 1)] = data_dict_org[case]
if is_more_big:
    save_obj(data_dict_extended,
             "dataset_info/train_dataset_info_EXACT09_LIDC_IDRI_crops_128_extended_" + "more_low_gen_" + str(
                 copy_times_I))
else:
    save_obj(data_dict_extended,
             "dataset_info/train_dataset_info_EXACT09_LIDC_IDRI_crops_128_extended_" + "more_high_gen_" + str(
                 copy_times_I))
    print("saved")

"""
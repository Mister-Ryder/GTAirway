import os
import numpy as np
import pickle
import edt

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

### Get pre-cropped LIDC-IDRI dataset info###
###
Precrop_dataset_for_train_path = "/home/ubuntu/shared-data/Precrop_dataset/Precrop_dataset_for_BAS"
Precrop_dataset_for_train_raw_path = Precrop_dataset_for_train_path + "/image"
Precrop_dataset_for_train_label_path = Precrop_dataset_for_train_path + "/label"
Precrop_dataset_for_train_skeleton_path = Precrop_dataset_for_train_path + "/skeleton"
Precrop_dataset_for_train_distance_path = Precrop_dataset_for_train_path + "/distance"
Precrop_dataset_for_train_gaussian_path = Precrop_dataset_for_train_path + "/gaussian"

skeleton_case_name_list = os.listdir(Precrop_dataset_for_train_skeleton_path)
distance_case_name_list = os.listdir(Precrop_dataset_for_train_distance_path)
gaussian_case_name_list = os.listdir(Precrop_dataset_for_train_gaussian_path)

assert skeleton_case_name_list == distance_case_name_list

data_dict = dict()

save_obj(data_dict, "dataset_info/3dataset_info_BAS_Skeleton_crops_128")

for idx, name in enumerate(skeleton_case_name_list):
    print("process " + str(name) + " | " + str(idx / len(skeleton_case_name_list)), end="\r")
    data_dict[name.split(".")[0]] = {}
    data_dict[name.split(".")[0]]["image"] = Precrop_dataset_for_train_raw_path + "/" + name
    data_dict[name.split(".")[0]]["label"] = Precrop_dataset_for_train_label_path + "/" + name
    data_dict[name.split(".")[0]]["skeleton"] = Precrop_dataset_for_train_skeleton_path + "/" + name
    data_dict[name.split(".")[0]]["distance"] = Precrop_dataset_for_train_distance_path + "/" + name
    data_dict[name.split(".")[0]]["gaussian"] = Precrop_dataset_for_train_gaussian_path + "/" + name

    # 加载CT图像数据
    image_temp = np.load(Precrop_dataset_for_train_raw_path + "/" + name.replace('_image_', '_'))
    label_temp = np.load(Precrop_dataset_for_train_label_path + "/" + name.replace('_label_','_'))

    data_dict[name.split(".")[0]]["airway_pixel_num"] = np.sum(label_temp)

    label_temp = np.array(label_temp, dtype=np.uint32, order='F')
    label_temp_edt = edt.edt(
        label_temp,
        black_border=True, order='F',
        parallel=1)

    data_dict[name.split(".")[0]]["airway_pixel_num_boundary"] = len(np.where(label_temp_edt == 1)[0])
    data_dict[name.split(".")[0]]["airway_pixel_num_inner"] = len(np.where(label_temp_edt > 1)[0])

    # 计算边界区域的体素强度
    boundary_region = (label_temp_edt == 1)
    boundary_image_values = image_temp[boundary_region]
    # 计算内部区域的体素强度
    inner_region = (label_temp_edt > 1)
    inner_image_values = image_temp[inner_region]
    # Define regions
    airway_region = (label_temp == 1)
    non_airway_region = (label_temp == 0)
    # 获取气道区域（标签为1的位置）
    # 计算气道区域的体素强度
    airway_image_values = image_temp[airway_region]  # 获取气道区域的体素强度

    # 计算体素强度异质性
    whole_strenghth_sum = np.sum(image_temp)
    whole_strenghth_mean = np.mean(image_temp)
    whole_strenghth_var = np.var(image_temp)
    non_airway_strength_mean = np.mean(image_temp[non_airway_region])
    non_airway_strength_var = np.var(image_temp[non_airway_region])
    airway_strength_sum = np.sum(airway_image_values)  # 气道区域的体素强度总和
    airway_strength_mean = np.mean(airway_image_values)  # 气道区域的体素强度均值
    airway_strength_var = np.var(airway_image_values)  # 气道区域的体素强度方差

    # 计算边界和内部体素的强度统计
    boundary_strength_sum = np.sum(boundary_image_values)
    inner_strength_sum = np.sum(inner_image_values)
    boundary_strength_mean = np.mean(boundary_image_values)
    inner_strength_mean = np.mean(inner_image_values)
    boundary_strength_var = np.var(boundary_image_values)
    inner_strength_var = np.var(inner_image_values)

    # Whole slice statistics
    data_dict[name.split(".")[0]]["whole_strength_sum"] = whole_strenghth_sum
    data_dict[name.split(".")[0]]["whole_strength_mean"] = whole_strenghth_mean
    data_dict[name.split(".")[0]]["whole_strength_var"] = whole_strenghth_var
    # Non-airway region statistics
    data_dict[name.split(".")[0]]["non_airway_strength_mean"] = non_airway_strength_mean
    data_dict[name.split(".")[0]]["non_airway_strength_var"] = non_airway_strength_var

    # airway region statistics
    data_dict[name.split(".")[0]]["airway_strength_sum"] = airway_strength_sum
    data_dict[name.split(".")[0]]["airway_strength_mean"] = airway_strength_mean
    data_dict[name.split(".")[0]]["airway_strength_var"] = airway_strength_var

    # 将边界和内部强度统计值保存到data_dict中
    data_dict[name.split(".")[0]]["boundary_strength_sum"] = boundary_strength_sum
    data_dict[name.split(".")[0]]["inner_strength_sum"] = inner_strength_sum
    data_dict[name.split(".")[0]]["boundary_strength_mean"] = boundary_strength_mean
    data_dict[name.split(".")[0]]["inner_strength_mean"] = inner_strength_mean
    data_dict[name.split(".")[0]]["boundary_strength_var"] = boundary_strength_var
    data_dict[name.split(".")[0]]["inner_strength_var"] = inner_strength_var
    data_dict[name.split(".")[0]]["boundary_inner_contrast"] = data_dict[name.split(".")[0]]["boundary_strength_mean"] - \
                                                               data_dict[name.split(".")[0]]["inner_strength_mean"]
    # Ratio and contrast metrics
    data_dict[name.split(".")[0]]["airway_non_airway_mean_ratio"] = data_dict[name.split(".")[0]]["airway_strength_mean"] / max(data_dict[name.split(".")[0]]["non_airway_strength_mean"], 1)

    print("process " + str(name) + " | " + str(idx / len(skeleton_case_name_list)))

save_obj(data_dict, "dataset_info/3dataset_info_BAS_Skeleton_crops_128")



'''
### Get the case names of LIDC-IDRI and BAS
### train训练集的info获取很快。

BAS_img_niigz_path = "/home/ubuntu/shared-data/BAS/train/image"
BAS_label_niigz_path = "/home/ubuntu/shared-data/BAS/train/label"

BAS_train_names = os.listdir(BAS_img_niigz_path)
BAS_train_names.sort()
BAS_train_label_names = os.listdir(BAS_label_niigz_path)
BAS_train_label_names.sort()
print(BAS_train_names, BAS_train_label_names)

BAS_train_names_rename = []
for BAS_name in BAS_train_names:
    BAS_train_names_rename.append("BAS_" + BAS_name.split(".")[0])
# BAS_train_names = np.array(BAS_train_names)
# BAS_train_names = np.unique(BAS_train_names)
print(BAS_train_names_rename)

train_names = BAS_train_names_rename
print("train names: " + str(train_names))

# BAS_img_niigz_path = "/home/ubuntu/shared-data/BAS/test/image"
# BAS_label_niigz_path = "/home/ubuntu/shared-data/BAS/test/label"

# BAS_test_names = os.listdir(BAS_img_niigz_path)
# BAS_test_names.sort()
# BAS_test_label_names = os.listdir(BAS_label_niigz_path)
# BAS_test_label_names.sort()
# print(BAS_test_names, BAS_test_label_names)

# BAS_test_names = []
# for BAS_name in BAS_test_names:
#     BAS_test_names.append("BAS_" + BAS_name.split(".")[0])
# BAS_test_names = np.array(BAS_test_names)
# BAS_test_names = np.unique(BAS_test_names)
#
# test_names = BAS_test_names
# print("train names: " + str(test_names))


### Get dataset dict for training

data_dict_BAS_128 = load_obj("dataset_info/dataset_info_BAS_Skeleton_crops_128")

train_test_set_dict_BAS_128 = {}
train_test_set_dict_BAS_128["train"] = {}
train_test_set_dict_BAS_128["test"] = {}

for case in data_dict_BAS_128.keys():
    if (case.split("_")[0] + "_" + case.split("_")[1]) in train_names:
        train_test_set_dict_BAS_128["train"][case] = data_dict_BAS_128[case]
    # elif (case.split("_")[0] + "_" + case.split("_")[1]) in test_names:
    #     train_test_set_dict_BAS_128["test"][case] = data_dict_BAS_128[case]

train_test_set_dict_BAS_128["train_names"] = train_names
# train_test_set_dict_BAS_128["test_names"] = test_names

# it has all info of images crops for training
### Get dataset_info for iterative training strategy. First, train with higher frequency on airways of high generations. Next, train with higher frequency on airways of low generations. And repeat the process.
data_dict_org = train_test_set_dict_BAS_128['train']

import copy

data_dict_extended = copy.deepcopy(data_dict_org)

file = open('output.txt', "w")
for idx, case in enumerate(data_dict_org.keys()):
    if data_dict_org[case]["airway_pixel_num"] > 0:
        if data_dict_org[case]["airway_pixel_num_inner"] == 0:
            copy_times_II = np.ceil(data_dict_org[case]["airway_pixel_num_boundary"])
            print(case, '[i=0]', data_dict_org[case]["airway_pixel_num_inner"],
                  data_dict_org[case]["airway_pixel_num_boundary"], copy_times_II, file=file)
        else:
            copy_times_II = np.ceil(
                data_dict_org[case]["airway_pixel_num_boundary"] / data_dict_org[case]["airway_pixel_num_inner"])
            print(case,'[b/i]',data_dict_org[case]["airway_pixel_num_inner"],data_dict_org[case]["airway_pixel_num_boundary"],copy_times_II,file=file)

        if copy_times_II>=45:
            for i in range(int(copy_times_II/2)):
                data_dict_extended[case + "_copy_" + str(i + 1)] = data_dict_org[case]
        if copy_times_II<=5:
            for i in range(2):
                data_dict_extended[case + "_copy_" + str(i + 1)] = data_dict_org[case]
        else:
            for i in range(1):
                data_dict_extended[case + "_copy_" + str(i + 1)] = data_dict_org[case]
file.close()

save_obj(data_dict_extended,
         "dataset_info/train_dataset_info_BAS_Skeleton_crops_128_extended_" + "generation_" + 'plus2')
'''
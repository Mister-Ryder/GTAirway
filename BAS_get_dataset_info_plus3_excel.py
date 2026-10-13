import os
import numpy as np
import pickle
import edt
import pandas as pd

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# 加载数据
data_dict = load_obj("dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_positions_and_ids")

# 转换为 DataFrame
df = pd.DataFrame.from_dict(data_dict, orient='index')

# 检查 'position' 是否在 DataFrame 列中
if 'position' not in df.columns:
    print("Warning: 'position' column not found in DataFrame.")
else:
    print("DataFrame columns:", df.columns)

# 对列名排序，并重新安排相关指标顺序
ordered_columns = [
    "image_size","position", "image_id", "block_id",  # 根据加载的pkl数据修改

    "airway_pixel_num",
    "airway_pixel_num_boundary",
    "airway_pixel_num_inner",

    "whole_strength_sum",
    "airway_strength_sum",
    "boundary_strength_sum",
    "inner_strength_sum",

    "whole_strength_mean",
    "airway_strength_mean",
    "boundary_strength_mean",
    "inner_strength_mean",
    "non_airway_strength_mean",

    "boundary_inner_ratio",
    "boundary_inner_contrast",  #["boundary_strength_mean"] - ["inner_strength_mean"]
    "airway_non_airway_mean_deta",

    "whole_strength_var",
    "airway_strength_var",
    "boundary_strength_var",
    "inner_strength_var",
    "non_airway_strength_var",

    "image", "label", "skeleton", "distance", "gaussian",

]

# 重新排列 DataFrame 的列
df = df[ordered_columns]

# 将 DataFrame 保存为 Excel 文件
df.to_excel("dataset_analysis_withposition.xlsx", index_label="Name")

print("Data successfully saved to dataset_analysis.xlsx")



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
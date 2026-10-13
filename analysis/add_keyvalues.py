# -*- coding: utf-8 -*-
# @Time    : 2025/3/12 19:19
# @Author  : JIA
# @FileName: add_keyvalues.py
# @Software: PyCharm
# @Blog    ：
import os
import pickle
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from collections import defaultdict
from pkl_to_excel import generate_excel_report


# 加载测试集元数据
# pkl_path = "dataset_info/3dataset_info_BAStest_merged.pkl"
# pkl_path = "dataset_info/3dataset_info_BAStrain_merged.pkl"
# pkl_path = "dataset_info/3dataset_info_BAStest_merged_dicescores.pkl"
pkl_path = "dataset_info/3dataset_info_BAStrain_merged_dicescores.pkl"



with open(pkl_path, 'rb') as f:
    metadata = pickle.load(f)

# 按病例ID分组元数据
case_dict = defaultdict(list)
for key in metadata:
    parts = key.split('_')
    case_id = parts[1]  # 假设键格式为 BAS_test_0066_0
    case_dict[case_id].append((key, parts[-1]))  # parts[-1]为block_id

# 遍历每个病例
for case_id in tqdm(case_dict, desc="补充指标"):
    # 处理每个切块
    for key, block_id in case_dict[case_id]:
        info = metadata[key]
        # 判断标签是否全为背景
        strength_ratio = info["boundary_inner_contrast"] / (abs(info["airway_non_airway_mean_deta"]) + 1e-6)
        airway_density = info["airway_pixel_num"] / (128 ** 3)

        metadata[key].update({
            'strength_ratio': strength_ratio,
            'airway_density': airway_density,
        })

# 保存修改后的元数据（覆盖原文件）
with open(pkl_path, 'wb') as f:  # 注意使用相同的路径
    pickle.dump(metadata, f)

# 统计报告
print(f"增加指标更新pkl已保存{pkl_path}")

generate_excel_report(pkl_path)
print("Excel generated successfully.")

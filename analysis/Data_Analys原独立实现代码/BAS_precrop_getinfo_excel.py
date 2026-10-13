import os
import numpy as np
import pickle
import edt
import pandas as pd

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# 加载数据
data_dict = load_obj("dataset_info/3dataset_info_BAS_merged")

# 转换为 DataFrame
df = pd.DataFrame.from_dict(data_dict, orient='index')

# 检查 'position' 是否在 DataFrame 列中
if 'position' not in df.columns:
    print("Warning: 'position' column not found in DataFrame.")
else:
    print("DataFrame columns:", df.columns)

# 对列名排序，并重新安排相关指标顺序
ordered_columns = [
    'image_path',  # 直接使用文件路径
    'label_path',
    'skeleton_path',
    'position',
    'image_size',
    'boundary_inner_ratio',
    'boundary_inner_contrast',
    'airway_non_airway_mean_deta',
    
    'airway_pixel_num',
    'airway_pixel_num_boundary',
    'airway_pixel_num_inner',
    'whole_strength_sum',
    'whole_strength_mean',
    'whole_strength_var',
    'non_airway_strength_mean',
    'non_airway_strength_var',
    'airway_strength_sum',
    'airway_strength_mean',
    'airway_strength_var',
    'boundary_strength_mean',
    'inner_strength_mean',

]

# 重新排列 DataFrame 的列
df = df[ordered_columns]

# 将 DataFrame 保存为 Excel 文件
df.to_excel("dataset_info/3dataset_info_BAS_merged.xlsx", index_label="Name")

print("Data successfully saved to dataset_analysis.xlsx")
# -*- coding: utf-8 -*-
# @Time    : 2025/3/12 10:54
# @Author  : JIA
# @FileName: pkl_to_excel.py
# @Software: PyCharm
# @Blog    ：
import pandas as pd
import pickle


def load_pkl_data(file_path):
    """统一加载pkl文件"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pkl_data(data, file_path):
    """保存pkl文件"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# =============================== Excel生成模块 ===============================
def generate_excel_report(data_path):
    """生成Excel报告（原BAS_precrop_getinfo_excel.py逻辑）"""
    # data_dict = load_pkl_data("dataset_info/3dataset_info_BAS_merged.pkl")
    data_dict = load_pkl_data(data_path)

    df = pd.DataFrame.from_dict(data_dict, orient='index')

    # ordered_columns = [
    #     'image_path', 'label_path', 'skeleton_path', 'position', 'image_size',
    #     'boundary_inner_ratio', 'boundary_inner_contrast', 'airway_non_airway_mean_deta',
    #     'airway_pixel_num', 'airway_pixel_num_boundary', 'airway_pixel_num_inner',
    #     'whole_strength_sum', 'whole_strength_mean', 'whole_strength_var',
    #     'non_airway_strength_mean', 'non_airway_strength_var',
    #     'airway_strength_sum', 'airway_strength_mean', 'airway_strength_var',
    #     'boundary_strength_mean', 'inner_strength_mean'
    # ]
    #
    # df = df[ordered_columns]
    df.to_excel(data_path.replace('pkl','xlsx'), index_label="Name")
    print("Excel generated successfully.")

data_path = "dataset_info/0/3dataset_info_BAStrain_merged_dicescores_wclusters_extended.pkl"
generate_excel_report(data_path)
print("Excel generated successfully.")
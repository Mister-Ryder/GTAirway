# -*- coding: utf-8 -*-
# @Time    : 2025/3/11 10:39
# @Author  : JIA
# @FileName: Crossdataset_Comparison.py.py
# @Software: PyCharm
# @Blog    ：
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_pkl_data(file_path):
    """加载pkl文件数据"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# 配置参数
# datasets = ["BRONATLASnew", , "AIIB23", "ATM22","BAStrain","BAStest"],"BRONATLASnew"
datasets = ["BAStest","AeroPath", "ATM22", "AIIB23","BAStrain",]
metrics = [


    # ("Airway Variance/Whole Variance",
    #  lambda data: abs(data["airway_strength_var"] / data["whole_strength_var"])
    #  if data["whole_strength_var"] != 0 else np.nan),

    ("boundary_inner_ratio",
     lambda data: data["boundary_inner_ratio"]),#显著

    ("strength_ratio",
     lambda data: data["boundary_inner_contrast"] / (abs(data["airway_non_airway_mean_deta"]) + 1e-6)),#显著

    ("airway_density",
     lambda data: data["airway_pixel_num"] / (128 ** 3)),  # 显著

    #底下的指标分布比较相似
    ("airway_non_airway_mean_deta",
     lambda data: data["airway_non_airway_mean_deta"]),

    ("boundary_inner_contrast",
     lambda data: data["boundary_inner_contrast"]),#显著

    ("Boundary Mean/Inner Mean",
     lambda data: abs(data["boundary_strength_mean"] / data["inner_strength_mean"])
     if data["inner_strength_mean"] != 0 else np.nan),
]

# 存储结构：字典嵌套字典
# {指标名称: {数据集名称: [数值列表]}}
metric_data = {name: {} for name, _ in metrics}

# 加载数据并计算指标
for dataset in datasets:
    file_path = f"dataset_info/3dataset_info_{dataset}_merged.pkl"
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, skipping...")
        continue

    data_dict = load_pkl_data(file_path)

    for metric_name, calculator in metrics:
        values = []
        for case_data in data_dict.values():
            value = calculator(case_data)
            if not np.isnan(value):  # 过滤NaN值
                values.append(value)

        # 对极端值进行Winsorize处理（保留98%的数据）
        if values:
            lower = np.percentile(values, 1)
            upper = np.percentile(values, 98)
            values = np.clip(values, lower, upper)

        metric_data[metric_name][dataset] = values

# 创建可视化图表
plt.figure(figsize=(25, 15))
plt.suptitle("Cross-Dataset Metrics Distribution Comparison", y=1.02, fontsize=20)

# 为每个指标创建子图
for idx, (metric_name, dataset_values) in enumerate(metric_data.items(), 1):
    ax = plt.subplot(2, 3, idx)

    # 绘制每个数据集的KDE曲线
    for dataset, values in dataset_values.items():
        if len(values) > 1:  # 需要至少两个点才能绘制KDE
            sns.kdeplot(values,
                        label=f"{dataset} (n={len(values)})",
                        linewidth=3,
                        alpha=0.2,
                        common_norm=False,
                        shade=True,
                        bw_adjust=0.7,
                        cut=3)  # 独立标准化

    # # 可选：手动设置x轴范围
    # all_values = []
    # for vals in dataset_values.values():
    #     all_values.extend(vals)
    # if all_values:
    #     min_val = np.min(all_values)
    #     max_val = np.max(all_values)
    #     padding = (max_val - min_val) * 0.1
    #     ax.set_xlim(min_val - padding, max_val + padding)
    #     if ax.get_xscale() == 'log':
    #         ax.set_xlim(max(min_val - padding, 1e-6), max_val + padding)

    # 设置对数坐标（针对特定指标）
    if "Variance" in metric_name or "ratio" in metric_name:
        ax.set_xscale('log')

    ax.set_title(metric_name, fontsize=20)
    ax.set_xlabel('',fontsize=15)
    ax.set_ylabel('density',fontsize=17)
    ax.tick_params(axis='both', which='major',labelsize=15)
    ax.legend(fontsize=15)

plt.tight_layout()
plt.savefig("report_image/cross_dataset_metrics_comparison2.png", bbox_inches='tight')
plt.close()

print("Cross-dataset comparison plot saved successfully.")
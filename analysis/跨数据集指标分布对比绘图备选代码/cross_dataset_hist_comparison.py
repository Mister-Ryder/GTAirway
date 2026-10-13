# -*- coding: utf-8 -*-
# @Time    : 2025/3/11 10:51
# @Author  : JIA
# @FileName: cross_dataset_hist_comparison.py.py
# @Software: PyCharm
# @Blog    ：
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator


def load_pkl_data(file_path):
    """加载pkl文件数据"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# 配置参数
datasets = ["BRONATLAStest", "BRONATLAStrain", "AeroPath", "AIIB23", "ATM22"]
palette = sns.color_palette("husl", n_colors=len(datasets))  # 固定颜色方案

metrics = [
    ("Boundary Mean/Inner Mean",
     lambda data: abs(data["boundary_strength_mean"] / data["inner_strength_mean"])
     if data["inner_strength_mean"] != 0 else np.nan,
     'log'),  # 新增坐标类型参数

    ("Airway Variance/Whole Variance",
     lambda data: abs(data["airway_strength_var"] / data["whole_strength_var"])
     if data["whole_strength_var"] != 0 else np.nan,
     'log'),

    ("airway_non_airway_mean_deta",
     lambda data: data["airway_non_airway_mean_deta"],
     'linear'),

    ("boundary_inner_contrast",
     lambda data: data["boundary_inner_contrast"],
     'linear'),

    ("strength_ratio",
     lambda data: data["boundary_inner_contrast"] / (abs(data["airway_non_airway_mean_deta"]) + 1e-6),
     'log'),

    ("airway_density",
     lambda data: data["airway_pixel_num"] / (128 ** 3),
     'linear')
]

# 存储结构：字典嵌套字典
metric_data = {name: {} for name, _, _ in metrics}

# 加载数据并计算指标
for dataset in datasets:
    file_path = f"dataset_info/3dataset_info_{dataset}_merged.pkl"
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, skipping...")
        continue

    data_dict = load_pkl_data(file_path)

    for metric_name, calculator, _ in metrics:
        values = []
        for case_data in data_dict.values():
            try:
                value = calculator(case_data)
                if np.isfinite(value):  # 过滤NaN和无穷值
                    values.append(float(value))
            except:
                continue

        # Winsorize处理（保留98%的数据）
        if len(values) > 10:
            lower = np.percentile(values, 1)
            upper = np.percentile(values, 99)
            values = np.clip(values, lower, upper)

        metric_data[metric_name][dataset] = values

# 创建可视化图表
plt.figure(figsize=(25, 18))
plt.suptitle("Cross-Dataset Metrics Histogram Comparison", y=1.02, fontsize=16)

# 为每个指标创建子图
for idx, (metric_name, _, scale_type) in enumerate(metrics, 1):
    ax = plt.subplot(2, 3, idx)

    # 计算全局bin范围
    all_values = []
    for values in metric_data[metric_name].values():
        all_values.extend(values)

    # 动态设置bins数量
    n_bins = min(50, int(len(all_values) / 10)) if len(all_values) > 100 else 20
    bins = np.histogram_bin_edges(all_values, bins=n_bins)

    # 绘制每个数据集的直方图
    for (dataset, values), color in zip(metric_data[metric_name].items(), palette):
        if len(values) > 1:
            sns.histplot(
                values,
                bins=bins,
                element="step",  # 阶梯式显示
                stat='density',
                alpha=0.4,
                color=color,
                label=f"{dataset} (n={len(values)})",
                linewidth=0  # 去除轮廓线
            )

    # 坐标轴设置
    ax.set_xscale(scale_type)
    ax.set_title(f"{metric_name}\nGlobal bins: {n_bins}", fontsize=12)
    ax.set_xlabel('Value Range')
    ax.set_ylabel('Density')

    # 对数坐标刻度优化
    if scale_type == 'log':
        ax.xaxis.set_major_locator(LogLocator(numticks=15))
        ax.xaxis.set_minor_locator(LogLocator(numticks=15, subs=np.arange(2, 10)))
        ax.tick_params(axis='x', which='minor', labelsize=6)

    # 添加辅助网格
    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    if scale_type == 'log':
        ax.grid(True, which='minor', linestyle=':', alpha=0.3)

    # 智能图例位置
    if idx % 3 == 1:  # 第一列显示图例
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.,
            frameon=False,
            fontsize=9
        )

plt.tight_layout(pad=3.0)
plt.savefig("report_image/cross_dataset_hist_comparison.png", bbox_inches='tight', dpi=300)
plt.close()

print("Histogram comparison plot saved successfully.")
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
from matplotlib.ticker import LogLocator, ScalarFormatter


def load_pkl_data(file_path):
    """加载pkl文件数据"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# 在原有代码基础上增加以下改进函数
def auto_axis_limits(values, scale_type, buffer_ratio=0.05):
    """智能坐标轴范围计算"""
    valid_vals = [v for v in values if not np.isnan(v)]
    if not valid_vals:
        return (0, 1)  # 默认范围

    # 动态范围计算
    min_val = np.min(valid_vals)
    max_val = np.max(valid_vals)

    # 防止0值影响对数坐标
    if scale_type == 'log':
        min_val = max(min_val, 1e-6) if min_val <= 0 else min_val
        buffer = 0  # 对数坐标自动扩展
    else:
        buffer = (max_val - min_val) * buffer_ratio

    return (min_val - buffer, max_val + buffer)

# 配置参数
# datasets = ["BRONATLASnew", "AeroPath", "AIIB23", "ATM22","BAStrain","BAStest"]
datasets = ["BRONATLASnew", "AIIB23", "ATM22","BAStrain"]
metrics = [
    ("Boundary Mean/Inner Mean",
     lambda data: abs(data["boundary_strength_mean"] / data["inner_strength_mean"])
     if data["inner_strength_mean"] != 0 else np.nan),

    ("Airway Variance/Whole Variance",
     lambda data: abs(data["airway_strength_var"] / data["whole_strength_var"])
     if data["whole_strength_var"] != 0 else np.nan),

    ("airway_non_airway_mean_deta",
     lambda data: data["airway_non_airway_mean_deta"]),

    ("boundary_inner_contrast",
     lambda data: data["boundary_inner_contrast"]),

    ("strength_ratio",
     lambda data: data["boundary_inner_contrast"] / (abs(data["airway_non_airway_mean_deta"]) + 1e-6)),

    ("airway_density",
     lambda data: data["airway_pixel_num"] / (128 ** 3))
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
            upper = np.percentile(values, 99)
            values = np.clip(values, lower, upper)

        metric_data[metric_name][dataset] = values

# 创建可视化图表
plt.figure(figsize=(25, 15))
plt.suptitle("Cross-Dataset Metrics Distribution Comparison", y=1.02, fontsize=16)

scale_type = 'log'

# 修改后的绘图部分
for idx, (metric_name, calculator) in enumerate(metrics, 1):
    ax = plt.subplot(2, 3, idx)

    # 收集所有数据集的值
    all_values = []
    for dataset in datasets:
        if dataset in metric_data[metric_name]:
            all_values.extend(metric_data[metric_name][dataset])

    # 动态设置坐标范围
    x_min, x_max = auto_axis_limits(all_values, scale_type)

    # 绘制每个数据集
    for dataset in datasets:
        if dataset in metric_data[metric_name]:
            values = metric_data[metric_name][dataset]
            sns.kdeplot(
                values,
                label=f"{dataset} (n={len(values)})",
                linewidth=2.5,
                alpha=0.85,
                common_norm=False,
                clip=(x_min, x_max)  # 关键改进：动态裁剪范围
            )

    # 高级坐标轴设置
    ax.set_xscale(scale_type)
    ax.set_xlim(x_min, x_max)

    # 优化对数坐标显示
    if scale_type == 'log':
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.tick_params(axis='x', which='both', labelsize=8, rotation=15)
    else:
        ax.tick_params(axis='x', which='major', labelsize=9)

    # 添加数据范围标注
    ax.annotate(f"Data range: {x_min:.2e} - {x_max:.2e}",
                xy=(0.98, 0.05), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))

plt.tight_layout()
plt.savefig("report_image/cross_dataset_metrics_comparison2.png", bbox_inches='tight')
plt.close()

print("Cross-dataset comparison plot saved successfully.")
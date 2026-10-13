# -*- coding: utf-8 -*-
# @Time    : 2025/3/11 10:56
# @Author  : JIA
# @FileName: cross_dataset_cdf_comparison.py.py
# @Software: PyCharm
# @Blog    ：
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns


def load_pkl_data(file_path):
    """加载pkl文件数据"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# 配置参数
datasets = ["BRONATLAStest", "BRONATLAStrain", "AeroPath", "AIIB23", "ATM22"]
colors = sns.color_palette("husl", n_colors=len(datasets))  # 固定颜色方案

metrics = [
    ("1. Boundary/Inner Mean Ratio",
     lambda data: abs(data["boundary_strength_mean"] / data["inner_strength_mean"])
     if data["inner_strength_mean"] != 0 else np.nan,
     'log'),  # 指标名称、计算函数、坐标类型

    ("2. Airway/Whole Var Ratio",
     lambda data: abs(data["airway_strength_var"] / data["whole_strength_var"])
     if data["whole_strength_var"] != 0 else np.nan,
     'log'),

    ("3. Airway-NonAirway Δμ",
     lambda data: data["airway_non_airway_mean_deta"],
     'linear'),

    ("4. Boundary-Inner Contrast",
     lambda data: data["boundary_inner_contrast"],
     'linear'),

    ("5. Contrast/Δμ Ratio",
     lambda data: data["boundary_inner_contrast"] / (abs(data["airway_non_airway_mean_deta"]) + 1e-6),
     'log'),

    ("6. Airway Density",
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

    for metric_name, calculator, scale_type in metrics:
        values = []
        for case_data in data_dict.values():
            try:
                value = calculator(case_data)
                if np.isfinite(value):  # 过滤NaN和无穷值
                    values.append(float(value))
            except:
                continue

        # 数据修剪（保留98%区间）
        if len(values) > 10:
            lower = np.percentile(values, 1)
            upper = np.percentile(values, 99)
            values = np.clip(values, lower, upper)

        metric_data[metric_name][dataset] = np.sort(values)

# 创建可视化图表
plt.figure(figsize=(22, 14))
plt.suptitle("Cross-Dataset Metrics Cumulative Distribution Comparison", y=1.0, fontsize=16)

# 为每个指标创建子图
for idx, (metric_name, _, scale_type) in enumerate(metrics, 1):
    ax = plt.subplot(2, 3, idx)

    # 绘制每个数据集的CDF曲线
    for (dataset, values), color in zip(metric_data[metric_name].items(), colors):
        if len(values) > 1:
            # 计算CDF
            x = np.sort(values)
            y = np.arange(1, len(x) + 1) / len(x)

            # 平滑处理
            if len(x) > 50:
                kernel = gaussian_kde(x)
                x_smooth = np.linspace(x.min(), x.max(), 500)
                y_smooth = np.interp(x_smooth, x, y)
                line = ax.plot(x_smooth, y_smooth, lw=2, color=color)
            else:
                line = ax.plot(x, y, 'o-', lw=1, markersize=3, color=color)

            # 标注关键分位数
            for q in [0.25, 0.5, 0.75]:
                q_val = np.quantile(values, q)
                ax.scatter(q_val, q, s=30, color=color, zorder=5,
                           edgecolor='white', linewidth=0.5)

    # 坐标轴设置
    ax.set_xscale(scale_type)
    ax.set_title(metric_name, fontsize=12, pad=12)
    ax.set_xlabel("Value", fontsize=9)
    ax.set_ylabel("Cumulative Probability", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 特殊坐标处理
    if scale_type == 'log':
        ax.set_xlim(left=np.percentile([v for vals in metric_data[metric_name].values()
                                        for v in vals], 1))

    # 添加图例
    if idx == 1:
        handles = [plt.Line2D([], [], color=color, label=ds)
                   for color, ds in zip(colors, datasets)]
        ax.legend(handles=handles, loc='lower right', fontsize=8,
                  bbox_to_anchor=(1.5, 0), frameon=False)

plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
plt.savefig("report_image/cross_dataset_cdf_comparison.png",
            bbox_inches='tight', dpi=300)
plt.close()

print("CDF comparison plot saved successfully.")
# -*- coding: utf-8 -*-
# @Time    : 2025/3/11 11:02
# @Author  : JIA
# @FileName: cross_dataset_hist_curve.py.py
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
datasets = ["BRONATLAStest", "BRONATLAStrain", "AeroPath", "AIIB23", "ATM22"]
colors = sns.color_palette("husl", n_colors=len(datasets))  # 固定颜色方案
line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # 差异化线型

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

    for metric_name, calculator, _ in metrics:
        values = []
        for case_data in data_dict.values():
            try:
                value = calculator(case_data)
                if np.isfinite(value):  # 过滤无效值
                    values.append(float(value))
            except:
                continue

        # 数据修剪（保留98%区间）
        if len(values) > 10:
            lower = np.percentile(values, 1)
            upper = np.percentile(values, 99)
            values = np.clip(values, lower, upper)

        metric_data[metric_name][dataset] = values

# 创建可视化图表
plt.figure(figsize=(25, 15))
plt.suptitle("Cross-Dataset Histogram Curve Comparison", y=1.02, fontsize=16)

# 为每个指标创建子图
for idx, (metric_name, _, scale_type) in enumerate(metrics, 1):
    ax = plt.subplot(2, 3, idx)

    # 计算全局bin范围
    all_values = []
    for values in metric_data[metric_name].values():
        all_values.extend(values)

    # 动态设置bins
    n_bins = min(30, int(len(all_values) ** 0.5))  # 基于数据量自动调整
    _, bin_edges = np.histogram(all_values, bins=n_bins)

    # 绘制每个数据集的曲线化直方图
    for (dataset, values), color, ls in zip(metric_data[metric_name].items(), colors, line_styles):
        if len(values) > 1:
            # 计算直方图
            hist, edges = np.histogram(values, bins=bin_edges, density=True)
            centers = (edges[:-1] + edges[1:]) / 2  # 计算bin中心

            # 曲线拟合（移动平均平滑）
            window_size = max(1, int(len(centers) * 0.1))
            weights = np.ones(window_size) / window_size
            smoothed = np.convolve(hist, weights, mode='same')

            # 绘制曲线
            ax.plot(centers, smoothed,
                    color=color,
                    linestyle=ls,
                    linewidth=2.5,
                    alpha=0.9,
                    label=f"{dataset} (n={len(values)})")

    # 坐标轴设置
    ax.set_xscale(scale_type)
    ax.set_title(metric_name, fontsize=12)
    ax.set_xlabel("Value Range")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3, linestyle=':')

    # 添加图例
    if idx == 1:
        ax.legend(bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  borderaxespad=0.,
                  frameon=False,
                  fontsize=9)

plt.tight_layout()
plt.savefig("report_image/cross_dataset_hist_curve.png",
            bbox_inches='tight', dpi=150)
plt.close()

print("Histogram curve comparison plot saved successfully.")
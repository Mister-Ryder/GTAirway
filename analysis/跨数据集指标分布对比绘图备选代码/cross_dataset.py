# -*- coding: utf-8 -*-
# @Time    : 2025/3/11 13:45
# @Author  : JIA
# @FileName: cross_dataset.py
# @Software: PyCharm
# @Blog    ：
# -*- coding: utf-8 -*-
# @Author  : JIA
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_pkl_data(file_path):
    """加载pkl文件"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def collect_cross_dataset_metrics(datasets):
    """收集跨数据集指标数据"""
    metric_data = []
    target_metrics = [
        "Boundary Mean/Inner Mean",
        "Airway Variance/Whole Variance",
        "airway_non_airway_mean_deta",
        "boundary_inner_contrast",
        "strength_ratio",
        "airway_density"
    ]

    for dataset in datasets:
        # 加载数据
        data_path = f"dataset_info/3dataset_info_{dataset}_merged.pkl"
        if not Path(data_path).exists():
            continue
        data_dict = load_pkl_data(data_path)

        # 计算指标
        metrics = {
            "Boundary Mean/Inner Mean": [
                abs(d["boundary_strength_mean"] / d["inner_strength_mean"])
                if d["inner_strength_mean"] != 0 else np.nan
                for d in data_dict.values()
            ],
            "Airway Variance/Whole Variance": [
                abs(d["airway_strength_var"] / d["whole_strength_var"])
                if d["whole_strength_var"] != 0 else np.nan
                for d in data_dict.values()
            ],
            "airway_non_airway_mean_deta": [d["airway_non_airway_mean_deta"] for d in data_dict.values()],
            "boundary_inner_contrast": [d["boundary_inner_contrast"] for d in data_dict.values()],
            "strength_ratio": [
                d["boundary_inner_contrast"] / (abs(d["airway_non_airway_mean_deta"]) + 1e-6)
                for d in data_dict.values()
            ],
            "airway_density": [d["airway_pixel_num"] / (128 ** 3) for d in data_dict.values()]
        }

        # 转换为DataFrame
        for metric_name in target_metrics:
            df = pd.DataFrame({
                'value': metrics[metric_name],
                'metric': metric_name,
                'dataset': dataset
            })
            metric_data.append(df.dropna())  # 过滤缺失值

    return pd.concat(metric_data, ignore_index=True)


def plot_cross_dataset_comparison(combined_df, save_path="cross_dataset_comparison.png"):
    """绘制跨数据集对比图"""
    plt.figure(figsize=(18, 24))
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    palette = sns.color_palette("husl", n_colors=len(combined_df['dataset'].unique()))

    # 为每个指标创建子图
    metrics = combined_df['metric'].unique()
    for idx, metric in enumerate(metrics, 1):
        ax = plt.subplot(3, 2, idx)
        subset = combined_df[combined_df['metric'] == metric]

        # 绘制KDE曲线
        sns.kdeplot(
            data=subset, x='value', hue='dataset',
            palette=palette, fill=True, alpha=0.3,
            linewidth=2, common_norm=False, ax=ax
        )

        # 美化图形
        ax.set_title(metric, fontsize=14, pad=12)
        ax.set_xlabel("")
        ax.legend(title='Dataset', title_fontsize=12,
                  fontsize=10, loc='upper right')

        # 自动调整坐标轴范围
        q_low = subset['value'].quantile(0.01)
        q_high = subset['value'].quantile(0.99)
        ax.set_xlim(q_low, q_high)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Cross-dataset comparison plot saved to {save_path}")


def plot_cross_dataset_comparison(combined_df, save_path="cross_dataset_comparison_hist.png"):
    """绘制跨数据集直方图对比"""
    plt.figure(figsize=(18, 24))
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    palette = sns.color_palette("husl", n_colors=len(combined_df['dataset'].unique()))

    # 为每个指标创建子图
    metrics = combined_df['metric'].unique()
    for idx, metric in enumerate(metrics, 1):
        ax = plt.subplot(3, 2, idx)
        subset = combined_df[combined_df['metric'] == metric]

        # 直方图关键参数
        bin_config = {
            "Boundary Mean/Inner Mean": np.linspace(0, 5, 51),
            "Airway Variance/Whole Variance": np.linspace(0, 3, 31),
            "airway_non_airway_mean_deta": np.linspace(-500, 1500, 51),
            "boundary_inner_contrast": np.linspace(-100, 300, 41),
            "strength_ratio": np.linspace(0, 1.5, 31),
            "airway_density": np.linspace(0, 0.15, 31)
        }

        # 绘制叠加直方图
        for dataset in subset['dataset'].unique():
            data = subset[subset['dataset'] == dataset]['value']
            sns.histplot(
                data,
                bins=bin_config[metric],
                alpha=0.4,
                edgecolor='black',
                linewidth=0.5,
                label=dataset,
                element='step',  # 增加阶梯状描边
                kde=False,  # 关闭KDE曲线
                palette=palette
            )

        # 美化图形
        ax.set_title(f"{metric}\n(Global Bins)", fontsize=12, pad=10)
        ax.set_xlabel("Value Range", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.legend(title='Dataset', fontsize=8,
                  title_fontsize=9, loc='upper right')

        # 自动调整坐标轴
        ax.set_xlim(bin_config[metric][0], bin_config[metric][-1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    target_datasets = ["BRONATLAStest", "BRONATLAStrain", "AeroPath", "AIIB23", "ATM22","BAStrain","BAStest"]
    combined_df = collect_cross_dataset_metrics(target_datasets)
    plot_cross_dataset_comparison(combined_df, "../report_image/cross_dataset_comparison.png")
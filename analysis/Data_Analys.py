# -*- coding: utf-8 -*-
# @Time    : 2025/3/10 16:20
# @Author  : JIA
# @FileName: Data_Analys.py
# @Software: PyCharm
# @Blog    ：
# -*- coding: utf-8 -*-
# @Author  : JIA
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering,DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns


# =============================== 通用工具函数 ===============================
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

    ordered_columns = [
        'image_path', 'label_path', 'skeleton_path', 'position', 'image_size',
        'boundary_inner_ratio', 'boundary_inner_contrast', 'airway_non_airway_mean_deta',
        'airway_pixel_num', 'airway_pixel_num_boundary', 'airway_pixel_num_inner',
        'whole_strength_sum', 'whole_strength_mean', 'whole_strength_var',
        'non_airway_strength_mean', 'non_airway_strength_var',
        'airway_strength_sum', 'airway_strength_mean', 'airway_strength_var',
        'boundary_strength_mean', 'inner_strength_mean'
    ]

    df = df[ordered_columns]
    df.to_excel(data_path.replace('pkl','xlsx'), index_label="Name")
    print("Excel report_image generated successfully.")


# =============================== 可视化模块 ===============================
def plot_heterogeneity_analysis(data_path,save_fig):
    """绘制异质性分析图（原BAS_precrop_getinfo_plotrelation+.py逻辑）"""
    # data_dict = load_pkl_data("dataset_info/3dataset_info_BAS_merged.pkl")
    data_dict = load_pkl_data(data_path)
    # 定义分析指标
    # structure_heterogeneity = [data["boundary_inner_ratio"] for data in data_dict.values()]#dice
    structure_heterogeneity = [data["dice"] for data in data_dict.values()]  #
    intensity_heterogeneities = {
        "Boundary Mean/Inner Mean": [
            abs(data["boundary_strength_mean"] / data["inner_strength_mean"])
            if data["inner_strength_mean"] != 0 else np.nan
            for data in data_dict.values()],
        "Airway Variance/Whole Variance": [
            abs(data["airway_strength_var"] / data["whole_strength_var"])
            if data["whole_strength_var"] != 0 else np.nan
            for data in data_dict.values()],
        "airway_non_airway_mean_deta": [data["airway_non_airway_mean_deta"] for data in data_dict.values()],
        "boundary_inner_ratio": [data["boundary_inner_ratio"] for data in data_dict.values()],
        "boundary_inner_contrast": [data["boundary_inner_contrast"] for data in data_dict.values()],
        # "strength_ratio": [data["boundary_inner_contrast"] / (abs(data["airway_non_airway_mean_deta"]) + 1e-6)
        #                    for data in data_dict.values()],
        "airway_density": [data["airway_pixel_num"] / (128 ** 3) for data in data_dict.values()]
    }

    # 绘制复合图表
    fig = plt.figure(figsize=(50, 20))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.4])

    for i, (name, values) in enumerate(intensity_heterogeneities.items()):
        # 直方图
        plt.subplot(gs[0, i])
        if name == "Boundary Mean/Inner Mean":
            # 设置横坐标范围
            plt.xlim(0, 1.5)
            values = [v for v in values if v <= 1.5 and v >= 0]
            # 设置纵坐标范围
            # plt.ylim(min_y_value, max_y_value)
        elif name == "airway_non_airway_mean_deta":
            # 设置横坐标范围
            plt.xlim(-1200, 1000)
            # 设置纵坐标范围
            # plt.ylim(min_y_value, max_y_value)
        elif name == "boundary_inner_contrast":
            # 设置横坐标范围
            plt.xlim(-400, 900)
            # # 设置纵坐标范围
            # plt.ylim(min_y_value, max_y_value)
        elif name == "Airway Variance/Whole Variance":
            # 设置横坐标范围
            plt.xlim(0, 1.4)
            # # 设置纵坐标范围
            # plt.ylim(min_y_value, max_y_value)

        plt.hist(values, bins=200, alpha=0.7, edgecolor='black')  #
        plt.title(f"Distribution of {name}")


        # 散点图和密度图
        filtered_data = [(s, v) for s, v in zip(structure_heterogeneity, values)
                         if np.isfinite(s) and np.isfinite(v)]
        if filtered_data:
            s_vals, i_vals = zip(*filtered_data)

            ax_scatter = fig.add_subplot(gs[1, i])
            ax_scatter.scatter(s_vals, i_vals, alpha=0.5)
            ax_scatter.set_title(f"Structure vs {name}")

            ax_hex = fig.add_subplot(gs[2, i])
            hb = ax_hex.hexbin(s_vals, i_vals, gridsize=30, cmap='coolwarm', mincnt=1)
            plt.colorbar(hb, ax=ax_hex, orientation='horizontal')

    plt.tight_layout()
    plt.savefig("report_image/heterogeneity_analysis_"+save_fig+".png")
    plt.close()
    print("Heterogeneity plots saved.")


# =============================== 聚类模块 ===============================
def prepare_features(data_dict):
    """特征预处理（原prepare_features函数）"""
    features, nan_keys = [], []
    for key in data_dict:
        data = data_dict[key]
        dice=data["dice"]
        ratio = data["boundary_inner_ratio"]
        # deta = data["airway_non_airway_mean_deta"]
        contrast = data["boundary_inner_contrast"]
        # chu = data["boundary_strength_mean"] / data["inner_strength_mean"]
        chu= data["boundary_inner_contrast"] / (abs(data["airway_non_airway_mean_deta"]) + 1e-6)
        des = data["airway_pixel_num"] / (128 ** 3)


        if not np.isnan(ratio):
            features.append([
                dice,
                ratio,
                chu,
                des,
                # contrast
                # (deta + 1000) / 1500 + (contrast + 50) / 350 + (chu - 0.6) / 0.6
            ])
        else:
            nan_keys.append(key)
    return np.array(features), nan_keys


def perform_clustering(features, n_clusters=20):
    """执行聚类分析"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # # DBSCAN示例
    dbscan = DBSCAN(eps=0.9, min_samples=100)
    clusters = dbscan.fit_predict(features_scaled)

    # model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=141)
    # clusters = model.fit_predict(features_scaled)
    return clusters, features_scaled


def visualize_clustering(features_scaled, clusters, nan_count):
    """可视化聚类结果"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_scaled)

    plt.figure(figsize=(12, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title(f"Clustering Result (NaN Samples: {nan_count})")
    plt.colorbar(label='Cluster')
    plt.savefig("_clustering_visualization.png")
    plt.close()


def clustering_analysis(data_path,save_fig):
    """执行完整聚类流程"""
    # data_dict = load_pkl_data("dataset_info/3dataset_info_BAS_merged.pkl")
    data_dict = load_pkl_data(data_path)

    features, nan_keys = prepare_features(data_dict)
    n_clusters=20
    # 聚类分析
    clusters, features_scaled = perform_clustering(features,n_clusters=n_clusters)
    print(f"Clustering metrics:")
    print(f"Silhouette Score: {silhouette_score(features_scaled, clusters):.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin_score(features_scaled, clusters):.4f}")

    # 保存带聚类标签的数据
    cluster_idx = 0
    for key in data_dict:
        if key in nan_keys:
            data_dict[key]["hierarchy_level"] = 0#"无内部气道"
        else:
            data_dict[key]["hierarchy_level"] = clusters[cluster_idx] + 1
            cluster_idx += 1
    save_pkl_data(data_dict, data_path.replace('.pkl',"_wclusters.pkl"))

    # # 可视化及统计
    # visualize_clustering(features_scaled, clusters, len(nan_keys))
    """可视化聚类结果"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_scaled)
    plt.figure(figsize=(12, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title(f"Clustering Result (NaN Samples: {len(nan_keys)})")
    plt.colorbar(label='Cluster')
    plt.savefig("report_image/clustering_visualization_"+save_fig+".png")
    plt.close()

    df = pd.DataFrame.from_dict(data_dict, orient='index')
    print("\nCluster distribution:")
    # 计算 hierarchy_level 列中各个类型的数量
    hierarchy_counts = df["hierarchy_level"].value_counts()
    print(df["hierarchy_level"].value_counts())
    n=hierarchy_counts.iat[0]//n_clusters
    target_count={}
    # 遍历每个 hierarchy_level 并重新分配数量
    for level, count in hierarchy_counts.items():
        # 计算当前 level 需要调整到的目标数量
        target_count[level] = round(n/count)
    print(target_count)

    import copy
    data_dict_extended = copy.deepcopy(data_dict)
    for key in data_dict:
        if key not in nan_keys:
            for i in range(target_count[data_dict[key]["hierarchy_level"]]):
                data_dict_extended[key+"_copy_"+str(i+1)]=data_dict[key]
    save_pkl_data(data_dict_extended, data_path.replace('.pkl', "_wclusters_extended.pkl"))

    df.to_excel(data_path.replace('.pkl','_wclusters.xlsx'), index_label="Name")


# =============================== 主程序 ===============================
if __name__ == "__main__":
    # dataset_name="AeroPath"
    # dataset_name="BAS"
    # dataset_name = "BRONATLAStrain"
    # dataset_name="BRONATLAStest","BRONATLAStrain","AeroPath","AIIB23","ATM22"

    for dataset_name in [
        "BAStrain",
        # "BAStest",
        # "BRONATLAStest",
        # "BRONATLAStrain",
        # "AeroPath",
        # "AIIB23",
        # "ATM22",
        # "BRONATLASnew"
    ]:
        data_path="dataset_info/3dataset_info_"+dataset_name+"_merged_dicescores.pkl"
        save_fig=dataset_name

        print(dataset_name)
        # # 执行顺序控制
        # print("=== 开始生成Excel报告 ===")
        # generate_excel_report(data_path)

        print("\n=== 开始异质性分析 ===")
        plot_heterogeneity_analysis(data_path,save_fig)

        print("\n=== 开始聚类分析 ===")
        clustering_analysis(data_path,save_fig)
        print("所有流程执行完成！\n")
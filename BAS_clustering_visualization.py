# -*- coding: utf-8 -*-
# @Time    : 2025/3/4 21:37
# @Author  : JIA
# @FileName: BAS_clustering_visualization.py.py
# @Software: PyCharm
# @Blog    ：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN,AgglomerativeClustering,SpectralClustering,AffinityPropagation,MeanShift
from sklearn.manifold import TSNE
import pickle
from sklearn.mixture import GaussianMixture


from sklearn.metrics import silhouette_score, davies_bouldin_score  # 导入评估指标


# 加载数据
def load_pkl_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# 保存数据
def save_pkl_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


# 处理NaN值并生成特征矩阵
def prepare_features(data_dict):
    features = []
    nan_keys = []
    for key in data_dict.keys():
        ratio = data_dict[key]["boundary_inner_ratio"]
        contrast = data_dict[key]["boundary_inner_contrast"]
        deta = data_dict[key]["airway_non_airway_mean_deta"]
        chu = data_dict[key]["boundary_strength_mean"]/data_dict[key]["inner_strength_mean"]
        if np.isnan(ratio):
            nan_keys.append(key)
        else:
            features.append([ratio,(deta+1000)/1500+(contrast+50)/350+(chu-0.6)/0.6])#,, ratio,contrast+chu+
    return np.array(features), nan_keys

# # 修改prepare_features函数
# def prepare_features(data_dict):
#     features, valid_keys = [], []
#     for key in data_dict.keys():
#         ratio = data_dict[key]["boundary_inner_ratio"]
#         if not np.isnan(ratio):
#             features.append([
#                 ratio,
#                 data_dict[key]["boundary_inner_contrast"],
#                 data_dict[key]["airway_non_airway_mean_deta"]
#             ])
#             valid_keys.append(key)
#     return np.array(features), valid_keys

# 聚类分析
def perform_clustering(features, n_clusters=3):
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # # 使用K-Means（可根据需求替换为DBSCAN）
    # kmeans = KMeans(n_clusters=n_clusters, random_state=141)#42
    # clusters = kmeans.fit_predict(features_scaled)
    # # # DBSCAN示例
    # dbscan = DBSCAN(eps=0.9, min_samples=100)
    # clusters = dbscan.fit_predict(features_scaled)
    # # 使用层次聚类（AgglomerativeClustering）
    # agglomerative = AgglomerativeClustering(n_clusters=n_clusters,  linkage='ward')#affinity='euclidean',
    # clusters = agglomerative.fit_predict(features_scaled)  # 聚类预测

    spectral = SpectralClustering(n_clusters=n_clusters,  affinity='nearest_neighbors', random_state=141)
    clusters = spectral.fit_predict(features_scaled)

    # gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    # clusters = gmm.fit_predict(features_scaled)

    # affinity = AffinityPropagation(random_state=42)
    # clusters = affinity.fit_predict(features_scaled)

    # mean_shift = MeanShift()
    # clusters = mean_shift.fit_predict(features_scaled)

    return clusters, features_scaled

# 聚类后评估指标
def evaluate_clustering(features_scaled, clusters):
    # 计算轮廓系数
    sil_score = silhouette_score(features_scaled, clusters)
    print(f"Silhouette Score: {sil_score}")

    # 计算Davies-Bouldin指数
    db_score = davies_bouldin_score(features_scaled, clusters)
    print(f"Davies-Bouldin Score: {db_score}")

# 可视化
def visualize_clustering(features_scaled, clusters, nan_count):
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_scaled)

    # 绘制聚类结果
    plt.figure(figsize=(12, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title(f"Clustering Result (Nan Samples: {nan_count})")
    plt.colorbar(label='Cluster')
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig("clustering_visualization.png")
    plt.close()


# 主程序
if __name__ == "__main__":
    # 加载数据
    data_dict = load_pkl_data("dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_positions_and_ids.pkl")

    # 准备特征矩阵并处理NaN
    features, nan_keys = prepare_features(data_dict)
    print(f"Total samples: {len(data_dict)}, NaN samples (无内部气道): {len(nan_keys)}")

    # 聚类分析
    clusters, features_scaled = perform_clustering(features, n_clusters=9)

    # 评估聚类效果
    evaluate_clustering(features_scaled, clusters)

    # 将层级标签写入data_dict
    cluster_idx = 0
    for key in data_dict.keys():
        if key in nan_keys:
            data_dict[key]["hierarchy_level"] = "无内部气道"  # 单独标记
        else:
            data_dict[key]["hierarchy_level"] = f"Level_{clusters[cluster_idx] + 1}"
            cluster_idx += 1

    # 保存更新后的数据
    save_pkl_data(data_dict, "dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_clusters.pkl")

    # 可视化
    visualize_clustering(features_scaled, clusters, len(nan_keys))

    # 生成统计报告
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df["hierarchy_level"] = df["hierarchy_level"].astype("category")
    print("\n层级分布统计:")
    print(df["hierarchy_level"].value_counts())

    # 保存为Excel（兼容原有Excel生成逻辑）
    df.to_excel("dataset_analysis_with_clusters.xlsx", index_label="Name")

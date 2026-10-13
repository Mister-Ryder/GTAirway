# -*- coding: utf-8 -*-
# @Time    : 2025/3/4 21:53
# @Author  : JIA
# @FileName: BAS_clustering_visualization2.py.py
# @Software: PyCharm
# @Blog    ：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import pickle

# 配置显示设置
pd.set_option('display.max_columns', 50)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_pkl_data(file_path):
    """加载预处理数据字典"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pkl_data(data, file_path):
    """保存更新后的数据"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def enhanced_features(data_dict):
    """生成增强特征矩阵"""
    features, nan_keys = [], []
    for key in data_dict:
        if np.isnan(data_dict[key]["boundary_inner_ratio"]):
            nan_keys.append(key)
            continue

        # 基础特征
        ratio = data_dict[key]["boundary_inner_ratio"]
        contrast = data_dict[key]["boundary_inner_contrast"]
        deta = data_dict[key]["airway_non_airway_mean_deta"]

        # 新增特征
        strength_ratio = contrast / (abs(deta) + 1e-6)  # 相对强度差异
        airway_density = data_dict[key]["airway_pixel_num"] / (128 ** 3)  # 气道密度

        features.append([ratio, contrast, deta, strength_ratio, airway_density])

    return np.array(features), nan_keys


def adaptive_clustering(features):
    """动态聚类策略"""
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 自动选择最优K值（2-4）
    best_k, max_score = 2, -1
    for k in range(2, 5):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(features_scaled)
        score = silhouette_score(features_scaled, kmeans.labels_)
        if score > max_score:
            best_k, max_score = k, score

    # 执行K-means聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)

    # 如果轮廓系数<0.4启用DBSCAN
    if max_score < 0.4:
        print("切换至密度聚类...")
        nn = NearestNeighbors(n_neighbors=5).fit(features_scaled)
        distances, _ = nn.kneighbors(features_scaled)
        eps = np.percentile(distances[:, -1], 95)
        clusters = DBSCAN(eps=eps, min_samples=50).fit_predict(features_scaled)

    return clusters, features_scaled


def merge_small_clusters(clusters, min_size=500):
    """自动合并小簇"""
    unique, counts = np.unique(clusters, return_counts=True)

    # 查找需要合并的簇
    small_clusters = unique[counts < min_size]
    if len(small_clusters) == 0:
        return clusters

    # 将小簇合并到最近的大簇
    cluster_centers = []
    for c in unique:
        cluster_centers.append(np.mean(features_scaled[clusters == c], axis=0))

    for sc in small_clusters:
        distances = np.linalg.norm(cluster_centers - cluster_centers[sc], axis=1)
        nearest_big_cluster = unique[np.argmax(distances)]
        clusters[clusters == sc] = nearest_big_cluster

    return clusters


def evaluate_clustering(features, labels):
    """聚类效果评估"""
    print("\n聚类质量评估:")
    print(f"轮廓系数: {silhouette_score(features, labels):.3f}")
    print(f"Calinski-Harabasz指数: {calinski_harabasz_score(features, labels):.1f}")
    print(f"Davies-Bouldin指数: {davies_bouldin_score(features, labels):.3f}")
    print("簇分布统计:", dict(zip(*np.unique(labels, return_counts=True))))


def visualize_results(features, features_scaled, clusters, nan_count):
    """可视化模块"""
    # t-SNE可视化
    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(features_scaled)

    plt.figure(figsize=(18, 6))

    # 子图1：聚类分布
    plt.subplot(121)
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1],
                          c=clusters, cmap='tab20', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(f"t-SNE可视化 (NaN样本数: {nan_count})")

    # 子图2：特征关联
    plt.subplot(122)
    sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=clusters,
                    palette='tab20', alpha=0.7)
    plt.xlabel("边界/内部比例")
    plt.ylabel("边界-内部对比度")
    plt.title("特征空间分布")

    plt.tight_layout()
    plt.savefig("optimized_clustering.png")
    plt.close()


if __name__ == "__main__":
    # 数据加载
    data_dict = load_pkl_data("dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_positions_and_ids.pkl")

    # 特征工程
    features, nan_keys = enhanced_features(data_dict)
    print(f"有效样本数: {len(features)}, 无效样本（无内部气道）: {len(nan_keys)}")

    # 聚类流程
    clusters, features_scaled = adaptive_clustering(features)
    clusters = merge_small_clusters(clusters, min_size=500)

    # 结果评估
    evaluate_clustering(features_scaled, clusters)

    # 结果保存
    cluster_idx = 0
    for key in data_dict:
        if key in nan_keys:
            data_dict[key]["hierarchy_level"] = "无内部气道"
        else:
            data_dict[key]["hierarchy_level"] = f"Level_{clusters[cluster_idx] + 1}"
            cluster_idx += 1

    save_pkl_data(data_dict, "dataset_info/3dataset_info_BAS_Skeleton_crops_128_optimized.pkl")

    # 可视化与报告
    visualize_results(features, features_scaled, clusters, len(nan_keys))

    # 生成统计报告
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df["hierarchy_level"] = df["hierarchy_level"].astype("category")

    print("\n最终层级分布:")
    level_stats = df["hierarchy_level"].value_counts().reset_index()
    level_stats.columns = ["层级", "样本数"]
    level_stats["占比"] = level_stats["样本数"] / len(data_dict) * 100
    print(level_stats.round(2))

    # 保存详细报告
    df.to_excel("optimized_dataset_analysis.xlsx", index_label="样本ID")
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the data dictionary
data_path = 'dataset_info/3dataset_info_BAS_merged.pkl'  # Replace with your actual path
data_dict = pickle.load(open(data_path, "rb"))

# Extract key statistics for heterogeneity analysis
structure_heterogeneity = [
    data["boundary_inner_ratio"]
    for data in data_dict.values()
] #结构强度

# Define intensity heterogeneity metrics with clear naming
intensity_heterogeneities = {
    "Boundary Mean/Inner Mean": [
        abs(data["boundary_strength_mean"] / data["inner_strength_mean"])
        if data["inner_strength_mean"] != 0 else np.nan
        for data in data_dict.values()
    ],
    # "Airway Mean/Non-Airway Mean": [
    #     data["airway_strength_mean"] / data["non_airway_strength_mean"]
    #     if data["airway_pixel_num_inner"] != 0 else np.nan
    #     for data in data_dict.values()
    # ],
    "Airway Variance/Whole Variance": [
        abs(data["airway_strength_var"] / data["whole_strength_var"])
        if data["whole_strength_var"] != 0 else np.nan
        for data in data_dict.values()
    ],
    # "Airway Mean/Whole Mean": [
    #     abs(data["airway_strength_mean"] / data["whole_strength_mean"])
    #     if data["whole_strength_mean"] != 0 else np.nan
    #     for data in data_dict.values()
    # ],

    #体素强度1
    "airway_non_airway_mean_deta": [
        data["airway_non_airway_mean_deta"]
        for data in data_dict.values()
    ],
    # 体素强度2
    "boundary_inner_contrast":[
        data["boundary_inner_contrast"] #"boundary_strength_mean" - "inner_strength_mean"
        for data in data_dict.values()
    ],
    #相对强度
    "strength_ratio":[
        data["boundary_inner_contrast"]/(abs(data["airway_non_airway_mean_deta"]) + 1e-6)
        for data in data_dict.values()
    ],
    # 气道密度
    "airway_density":[
        data["airway_pixel_num"] / (128 ** 3)
        for data in data_dict.values()
    ]
}

#打印字典的某个key值
# print(intensity_heterogeneities["Airway Mean/Non-Airway Mean"])

# Plot histograms for all heterogeneity metrics
plt.figure(figsize=(50, 7.5))
for i, (name, values) in enumerate(intensity_heterogeneities.items(), start=1):
    plt.subplot(1, 6, i)
    plt.hist(values, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {name}")
    plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter plots of Structure Heterogeneity vs. each Intensity Heterogeneity
# plt.figure(figsize=(35, 10))
# 创建一个新的 figure 和 gridspec 布局
fig = plt.figure(figsize=(50, 20))  # 调整整个图形的高度
gs = fig.add_gridspec(3, 7, height_ratios=[1,1,1.4])  # 保持上面和下面两行的比例相同，给colorbar留出空间
for i, (name, values) in enumerate(intensity_heterogeneities.items(), start=1):

    plt.subplot(gs[0, i - 1])
    plt.hist(values, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {name}")
    plt.grid(True)

    # Filter valid data for scatter plot
    filtered_data = [
        (s, i_val) for s, i_val in zip(structure_heterogeneity, values)
        if np.isfinite(s) and np.isfinite(i_val)
    ]
    if filtered_data:  # Ensure there are valid data points
        structure_heterogeneity_filtered, intensity_heterogeneity_filtered = zip(*filtered_data)
        # 绘制散点图（上行）
        ax1 = fig.add_subplot(gs[1, i - 1])  # 设置上行的子图
        ax1.scatter(structure_heterogeneity_filtered, intensity_heterogeneity_filtered, alpha=0.5)
        ax1.set_xlabel("Structure Heterogeneity")
        ax1.set_ylabel(name)
        ax1.set_title(f"Scatter Plot: Structure vs ({name})")
        ax1.grid(True)

        # 绘制密度热图（下行）
        ax2 = fig.add_subplot(gs[2, i - 1])  # 设置下行的子图
        hb = ax2.hexbin(
            structure_heterogeneity_filtered,
            intensity_heterogeneity_filtered,
            gridsize=30,
            cmap='coolwarm',
            mincnt=1
        )
        ax2.set_xlabel("Structure Heterogeneity")
        ax2.set_ylabel(name)
        ax2.set_title(f"Density Heatmap: Structure vs {name}")
        ax2.grid(True)

        cb = fig.colorbar(hb, ax=ax2, orientation='horizontal')
        cb.set_label('Density')

plt.tight_layout()
plt.show()

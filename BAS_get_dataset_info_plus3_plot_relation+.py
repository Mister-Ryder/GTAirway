import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the data dictionary
data_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128.pkl'  # Replace with your actual path
data_dict = pickle.load(open(data_path, "rb"))

# Extract key statistics for heterogeneity analysis
structure_heterogeneity = [
    data["airway_pixel_num_boundary"] / data["airway_pixel_num_inner"]
    if data["airway_pixel_num_inner"] > 0 else np.nan
    for data in data_dict.values()
] #结构强度

# Define intensity heterogeneity metrics with clear naming
intensity_heterogeneities = {
    # "Boundary Mean/Inner Mean": [
    #     abs(data["boundary_strength_mean"] / data["inner_strength_mean"])
    #     if data["inner_strength_mean"] != 0 else np.nan
    #     for data in data_dict.values()
    # ],
    # "Airway Mean/Non-Airway Mean": [
    #     data["airway_strength_mean"] / data["non_airway_strength_mean"]
    #     if data["airway_pixel_num_inner"] != 0 else np.nan
    #     for data in data_dict.values()
    # ],
    # "Airway Variance/Whole Variance": [
    #     abs(data["airway_strength_var"] / data["whole_strength_var"])
    #     if data["whole_strength_var"] != 0 else np.nan
    #     for data in data_dict.values()
    # ],
    # "Airway Mean/Whole Mean": [
    #     abs(data["airway_strength_mean"] / data["whole_strength_mean"])
    #     if data["whole_strength_mean"] != 0 else np.nan
    #     for data in data_dict.values()
    # ],
    # "Airway Variance/Airway Mean": [
    #     abs(data["airway_strength_var"] / data["airway_strength_mean"])
    #     if data["airway_strength_mean"] != 0 else np.nan
    #     for data in data_dict.values()
    # ],
    "Intensity Contrast (Foreground-Background)": [
        data["airway_strength_mean"] - data["non_airway_strength_mean"]
        for data in data_dict.values()
    ], #体素强度1
    # "Airway Variance-Non-Airway Variance": [
    #     data["airway_strength_var"] / data["non_airway_strength_var"]
    #     for data in data_dict.values()
    # ],
    "Intensity Contrast (Airway Boundary-Inner)":[
        data["boundary_inner_contrast"] #"boundary_strength_mean" - "inner_strength_mean"
        for data in data_dict.values()
    ], #体素强度2
    # "intensity_heterogeneities":[
    #     0.5*(5*(data["boundary_inner_contrast"]+50)) + 0.5*(data["airway_strength_mean"] - data["non_airway_strength_mean"]+1000)
    #     for data in data_dict.values()
    # ]

}

# Plot histograms for all heterogeneity metrics
plt.figure(figsize=(50, 7.5))
for i, (name, values) in enumerate(intensity_heterogeneities.items(), start=1):
    plt.subplot(1, 7, i)
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
fig = plt.figure(figsize=(15, 20))  # 调整整个图形的高度
gs = fig.add_gridspec(3, 2, height_ratios=[1,1,1.4])  # 保持上面和下面两行的比例相同，给colorbar留出空间
for i, (name, values) in enumerate(intensity_heterogeneities.items(), start=1):

    # plt.subplot(gs[0, i - 1])
    # plt.hist(values, bins=30, alpha=0.7, edgecolor='black')
    # plt.xlabel(name)
    # plt.ylabel("Frequency")
    # plt.title(f"Distribution of {name}")
    # plt.grid(True)

    # Filter valid data for scatter plot
    filtered_data = [
        (s, i_val) for s, i_val in zip(structure_heterogeneity, values)
        if np.isfinite(s) and np.isfinite(i_val)
    ]
    if filtered_data:  # Ensure there are valid data points
        structure_heterogeneity_filtered, intensity_heterogeneity_filtered = zip(*filtered_data)
        # # 绘制散点图（上行）
        # ax1 = fig.add_subplot(gs[1, i - 1])  # 设置上行的子图
        # ax1.scatter(structure_heterogeneity_filtered, intensity_heterogeneity_filtered, alpha=0.5)
        # ax1.set_xlabel("Structure Heterogeneity")
        # ax1.set_ylabel(name)
        # ax1.set_title(f"Scatter Plot: Structure vs ({name})")
        # ax1.grid(True)

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

# # Filter valid data for plotting
# filtered_data = [
#     (s, a, b) for s, a, b in zip(structure_heterogeneity,
#                                   intensity_heterogeneities["Airway Mean-Non-Airway Mean"],
#                                   intensity_heterogeneities["boundary_inner_contrast"])
#     if np.isfinite(s) and np.isfinite(a) and np.isfinite(b)
# ]
#
# structure_heterogeneity_filtered, airway_non_airway_filtered, boundary_inner_filtered = zip(*filtered_data)
#
# # Create a 3D scatter plot
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the scatter points in 3D
# sc = ax.scatter(structure_heterogeneity_filtered, airway_non_airway_filtered, boundary_inner_filtered,
#                 c=structure_heterogeneity_filtered, cmap='viridis', alpha=0.7, edgecolors='w', s=100)
#
# # Label axes
# ax.set_xlabel("Structure Heterogeneity")
# ax.set_ylabel("Intensity Contrast (Foreground-Background)")
# ax.set_zlabel("Intensity Contrast (Airway Boundary-Inner)")
# ax.set_title("3D Distribution of Heterogeneity Metrics")
#
# # Add color bar to represent density or value
# cbar = fig.colorbar(sc)
# cbar.set_label("Structure Heterogeneity")
#
# # Show the plot
# plt.show()

# import pandas as pd
#
# # Combine data into a DataFrame
# data_for_excel = {
#     "Structure Heterogeneity": structure_heterogeneity,
#     "Airway Mean-Non-Airway Mean": intensity_heterogeneities["Airway Mean-Non-Airway Mean"],
#     "Boundary Inner Contrast": intensity_heterogeneities["boundary_inner_contrast"],
# }
#
# # Convert the dictionary to a DataFrame
# df = pd.DataFrame(data_for_excel)
#
# # Filter out rows with NaN values
# df_cleaned = df.dropna()
#
# # Save the cleaned DataFrame to an Excel file
# excel_file_path = 'heterogeneity_data.xlsx'
# df_cleaned.to_excel(excel_file_path, index=False)
#
# # Print success message
# print(f"Data has been saved to {excel_file_path}")
#
# import pickle
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy import stats
# import seaborn as sns
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.linear_model import LinearRegression
#
# # Set style for better visualizations
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_context("paper", font_scale=1.2)
#
# # Load the data dictionary
# data_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128.pkl'
# data_dict = pickle.load(open(data_path, "rb"))
#
# # Extract key statistics for heterogeneity analysis - Structure Heterogeneity
# structure_heterogeneity = [
#     data["airway_pixel_num_boundary"] / data["airway_pixel_num_inner"]
#     if data["airway_pixel_num_inner"] > 0 else np.nan
#     for data in data_dict.values()
# ]
#
# # Define intensity heterogeneity metrics with clear naming
# intensity_heterogeneities = {
#     "Intensity Contrast (Airway vs Non-Airway)": [
#         data["airway_strength_mean"] - data["non_airway_strength_mean"]
#         for data in data_dict.values()
#     ],
#     "Boundary-Inner Contrast": [
#         data["boundary_inner_contrast"]
#         for data in data_dict.values()
#     ]
# }
#
# # Filter valid data for analysis
# valid_indices = [i for i, (s, a, b) in enumerate(zip(
#     structure_heterogeneity,
#     intensity_heterogeneities["Intensity Contrast (Airway vs Non-Airway)"],
#     intensity_heterogeneities["Boundary-Inner Contrast"]
# )) if np.isfinite(s) and np.isfinite(a) and np.isfinite(b)]
#
# filtered_data = {
#     "Structure Heterogeneity": [structure_heterogeneity[i] for i in valid_indices],
#     "Intensity Contrast (Airway vs Non-Airway)": [
#         intensity_heterogeneities["Intensity Contrast (Airway vs Non-Airway)"][i] for i in valid_indices],
#     "Boundary-Inner Contrast": [intensity_heterogeneities["Boundary-Inner Contrast"][i] for i in valid_indices]
# }
#
# # Create DataFrame for statistical analysis
# df = pd.DataFrame(filtered_data)
#
# # 1. Plot histograms of heterogeneity distributions
# plt.figure(figsize=(15, 5))
#
# # Structure heterogeneity histogram with KDE
# plt.subplot(1, 3, 1)
# sns.histplot(filtered_data["Structure Heterogeneity"], bins=30, kde=True, color='steelblue')
# plt.xlabel("Structure Heterogeneity")
# plt.ylabel("Frequency")
# plt.title("Distribution of Structure Heterogeneity")
# plt.grid(True)
#
# # Intensity heterogeneity histograms with KDE
# metric_colors = ['indianred', 'darkgreen']
# for i, (name, values) in enumerate(intensity_heterogeneities.items(), start=2):
#     plt.subplot(1, 3, i)
#     sns.histplot(filtered_data[name], bins=30, kde=True, color=metric_colors[i - 2])
#     plt.xlabel(name)
#     plt.ylabel("Frequency")
#     plt.title(f"Distribution of {name}")
#     plt.grid(True)
#
# plt.tight_layout()
# plt.savefig("heterogeneity_distributions.png", dpi=300)
# plt.show()
#
# # 2. Create correlation plots between structure and intensity heterogeneities
# fig = plt.figure(figsize=(15, 12))
# gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
#
# # Scatter plots
# for i, (name, color) in enumerate(zip(intensity_heterogeneities.keys(), ['indianred', 'darkgreen'])):
#     # Scatter plot with regression line
#     ax1 = fig.add_subplot(gs[0, i])
#     ax1.scatter(filtered_data["Structure Heterogeneity"], filtered_data[name], alpha=0.5, color=color)
#
#     # Add regression line
#     slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_data["Structure Heterogeneity"],
#                                                                    filtered_data[name])
#     x_line = np.linspace(min(filtered_data["Structure Heterogeneity"]), max(filtered_data["Structure Heterogeneity"]),
#                          100)
#     y_line = slope * x_line + intercept
#     ax1.plot(x_line, y_line, 'k--', linewidth=2)
#
#     ax1.set_xlabel("Structure Heterogeneity")
#     ax1.set_ylabel(name)
#     ax1.set_title(f"Scatter Plot: Structure vs {name}")
#     ax1.grid(True)
#
#     # Calculate correlation and add text annotation
#     corr, p_value = stats.pearsonr(filtered_data["Structure Heterogeneity"], filtered_data[name])
#     ax1.annotate(f"Pearson r: {corr:.3f}\np-value: {p_value:.3e}",
#                  xy=(0.05, 0.95), xycoords='axes fraction',
#                  ha='left', va='top',
#                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
#
#     # Density heatmap
#     ax2 = fig.add_subplot(gs[1, i])
#     hb = ax2.hexbin(
#         filtered_data["Structure Heterogeneity"],
#         filtered_data[name],
#         gridsize=30,
#         cmap='viridis',
#         mincnt=1
#     )
#     ax2.set_xlabel("Structure Heterogeneity")
#     ax2.set_ylabel(name)
#     ax2.set_title(f"Density Heatmap: Structure vs {name}")
#     ax2.grid(True)
#
#     cb = fig.colorbar(hb, ax=ax2, orientation='horizontal')
#     cb.set_label('Density')
#
# plt.tight_layout()
# plt.savefig("structure_intensity_correlations.png", dpi=300)
# plt.show()
#
# # 3. Relationship between the two intensity heterogeneity metrics
# plt.figure(figsize=(8, 6))
#
# # Scatter plot with regression line
# plt.scatter(filtered_data["Intensity Contrast (Airway vs Non-Airway)"],
#             filtered_data["Boundary-Inner Contrast"],
#             alpha=0.7, color='purple')
#
# # Add regression line
# slope, intercept, r_value, p_value, std_err = stats.linregress(
#     filtered_data["Intensity Contrast (Airway vs Non-Airway)"],
#     filtered_data["Boundary-Inner Contrast"]
# )
# x_line = np.linspace(
#     min(filtered_data["Intensity Contrast (Airway vs Non-Airway)"]),
#     max(filtered_data["Intensity Contrast (Airway vs Non-Airway)"]),
#     100
# )
# y_line = slope * x_line + intercept
# plt.plot(x_line, y_line, 'k--', linewidth=2)
#
# # Add correlation statistics
# plt.annotate(f"Pearson r: {r_value:.3f}\np-value: {p_value:.3e}\nSlope: {slope:.3f}",
#              xy=(0.05, 0.95), xycoords='axes fraction',
#              ha='left', va='top',
#              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
#
# plt.xlabel("Intensity Contrast (Airway vs Non-Airway)")
# plt.ylabel("Boundary-Inner Contrast")
# plt.title("Relationship Between Intensity Heterogeneity Metrics")
# plt.grid(True)
#
# plt.tight_layout()
# plt.savefig("intensity_metrics_relationship.png", dpi=300)
# plt.show()
#
# # 4. 3D visualization of all three metrics
# fig = plt.figure(figsize=(12, 10))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the scatter points in 3D with color representing structure heterogeneity
# sc = ax.scatter(
#     filtered_data["Structure Heterogeneity"],
#     filtered_data["Intensity Contrast (Airway vs Non-Airway)"],
#     filtered_data["Boundary-Inner Contrast"],
#     c=filtered_data["Structure Heterogeneity"],
#     cmap='viridis',
#     alpha=0.7,
#     edgecolors='w',
#     s=100
# )
#
# # Add regression plane
# X = np.column_stack((
#     filtered_data["Structure Heterogeneity"],
#     filtered_data["Intensity Contrast (Airway vs Non-Airway)"]
# ))
# y = filtered_data["Boundary-Inner Contrast"]
# model = LinearRegression().fit(X, y)
# r_sq = model.score(X, y)
#
# # Create a meshgrid to plot the plane
# s_min, s_max = min(filtered_data["Structure Heterogeneity"]), max(filtered_data["Structure Heterogeneity"])
# a_min, a_max = min(filtered_data["Intensity Contrast (Airway vs Non-Airway)"]), max(
#     filtered_data["Intensity Contrast (Airway vs Non-Airway)"])
# s_grid, a_grid = np.meshgrid(np.linspace(s_min, s_max, 20), np.linspace(a_min, a_max, 20))
# X_pred = np.column_stack((s_grid.ravel(), a_grid.ravel()))
# b_pred = model.predict(X_pred)
# b_grid = b_pred.reshape(s_grid.shape)
#
# # Plot the regression plane
# surf = ax.plot_surface(s_grid, a_grid, b_grid, alpha=0.3, color='red')
#
# # Label axes
# ax.set_xlabel("Structure Heterogeneity")
# ax.set_ylabel("Intensity Contrast (Airway vs Non-Airway)")
# ax.set_zlabel("Boundary-Inner Contrast")
# ax.set_title(f"3D Distribution of Heterogeneity Metrics\nMultiple R²: {r_sq:.3f}")
#
# # Add color bar
# cbar = fig.colorbar(sc)
# cbar.set_label("Structure Heterogeneity")
#
# plt.tight_layout()
# plt.savefig("heterogeneity_3d_distribution.png", dpi=300)
# plt.show()
#
# # 5. Calculate and print statistical metrics to support the paper introduction
# print("\n--- Statistical Summary ---")
#
# # Basic descriptive statistics for all metrics
# for metric in filtered_data.keys():
#     print(f"{metric}:")
#     print(f"  Mean ± SD: {np.mean(filtered_data[metric]):.3f} ± {np.std(filtered_data[metric]):.3f}")
#     print(
#         f"  Median [IQR]: {np.median(filtered_data[metric]):.3f} [{np.percentile(filtered_data[metric], 25):.3f}, {np.percentile(filtered_data[metric], 75):.3f}]")
#     print(f"  Range: [{min(filtered_data[metric]):.3f}, {max(filtered_data[metric]):.3f}]")
#     print(f"  Skewness: {stats.skew(filtered_data[metric]):.3f}")
#     print(f"  Kurtosis: {stats.kurtosis(filtered_data[metric]):.3f}")
#     print()
#
# # Coefficient of variation (measure of relative dispersion)
# print("\n--- Heterogeneity Dispersion Metrics ---")
# for metric in filtered_data.keys():
#     cv = np.std(filtered_data[metric]) / np.mean(filtered_data[metric])
#     print(f"Coefficient of Variation ({metric}): {cv:.3f}")
#
# # Statistical tests for distribution non-uniformity
# print("\n--- Distribution Non-Uniformity Tests ---")
# for metric in filtered_data.keys():
#     # Shapiro-Wilk test for normality
#     shapiro_result = stats.shapiro(filtered_data[metric])
#     print(f"{metric} (Shapiro-Wilk): W={shapiro_result.statistic:.3f}, p-value={shapiro_result.pvalue:.3e}")
#
#     # KS test against uniform distribution
#     val_min, val_max = min(filtered_data[metric]), max(filtered_data[metric])
#     ks_result = stats.kstest(filtered_data[metric], 'uniform', args=(val_min, val_max - val_min))
#     print(f"{metric} (KS vs Uniform): D={ks_result.statistic:.3f}, p-value={ks_result.pvalue:.3e}")
#
# # Correlation analysis between metrics
# print("\n--- Correlation Analysis ---")
# # Structure vs Intensity Correlations
# for name in ["Intensity Contrast (Airway vs Non-Airway)", "Boundary-Inner Contrast"]:
#     pearson_r, pearson_p = stats.pearsonr(filtered_data["Structure Heterogeneity"], filtered_data[name])
#     spearman_r, spearman_p = stats.spearmanr(filtered_data["Structure Heterogeneity"], filtered_data[name])
#     print(f"Structure Heterogeneity vs {name}:")
#     print(f"  Pearson: r={pearson_r:.3f}, p-value={pearson_p:.3e}")
#     print(f"  Spearman: rho={spearman_r:.3f}, p-value={spearman_p:.3e}")
#
# # Correlation between intensity metrics
# pearson_r, pearson_p = stats.pearsonr(
#     filtered_data["Intensity Contrast (Airway vs Non-Airway)"],
#     filtered_data["Boundary-Inner Contrast"]
# )
# spearman_r, spearman_p = stats.spearmanr(
#     filtered_data["Intensity Contrast (Airway vs Non-Airway)"],
#     filtered_data["Boundary-Inner Contrast"]
# )
# print(f"Intensity Contrast (Airway vs Non-Airway) vs Boundary-Inner Contrast:")
# print(f"  Pearson: r={pearson_r:.3f}, p-value={pearson_p:.3e}")
# print(f"  Spearman: rho={spearman_r:.3f}, p-value={spearman_p:.3e}")
#
# # Save data to Excel for further analysis
# df.to_excel('heterogeneity_analysis.xlsx', index=False)
# print(f"Data saved to 'heterogeneity_analysis.xlsx' with {len(df)} valid samples")

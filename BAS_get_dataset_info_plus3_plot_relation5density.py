# Re-import necessary libraries
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Reload the cleaned dataset
data_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128.pkl'
data_dict = pickle.load(open(data_path, "rb"))

# Extract relevant metrics
structure_heterogeneity = [
    data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1)
    for data in data_dict.values()
]

Intensity_heterogeneity_1 = [
    data["boundary_strength_mean"] / max(data["inner_strength_mean"], 1)
    for data in data_dict.values()
]
Intensity_heterogeneity_2 = [
    data["airway_strength_mean"] / max(data["non_airway_strength_mean"], 1)
    for data in data_dict.values()
]

Intensity_heterogeneity_3 = [
    data["airway_strength_var"] / max(data["whole_strength_var"], 1)
    for data in data_dict.values()
]

Intensity_heterogeneity_4 = [
    data["airway_strength_mean"] / max(data["whole_strength_mean"], 1)
    for data in data_dict.values()
]

Intensity_heterogeneity_5 = [
    data["airway_strength_var"] / max(data["whole_strength_mean"], 1)
    for data in data_dict.values()
]


data_cleaned = pd.DataFrame({
    "Structure Heterogeneity": structure_heterogeneity,
    "Whole Strength Mean": [data["whole_strength_mean"] for data in data_dict.values()],
    "Non-Airway Strength Mean": [data["non_airway_strength_mean"] for data in data_dict.values()],
    "Airway Strength Mean": [data["airway_strength_mean"] for data in data_dict.values()],
    "Boundary-Inner Contrast": [data["boundary_inner_contrast"] for data in data_dict.values()],
    "Airway/Non-Airway Mean Ratio": [data["airway_non_airway_mean_ratio"] for data in data_dict.values()],
    "Whole Strength Variance": [data["whole_strength_var"] for data in data_dict.values()],
}).replace([np.inf, -np.inf], np.nan).dropna()

data_cleaned2 = pd.DataFrame({
    "Structure Heterogeneity": structure_heterogeneity,
    "Intensity Heterogeneity1": Intensity_heterogeneity_1,
    "Intensity Heterogeneity2": Intensity_heterogeneity_2,
    "Intensity Heterogeneity3": Intensity_heterogeneity_3,
    "Intensity Heterogeneity4": Intensity_heterogeneity_4,
    "Intensity Heterogeneity5": Intensity_heterogeneity_5,
}).replace([np.inf, -np.inf], np.nan).dropna()


composite_metric_layered = (
    0.5 * data_cleaned2["Intensity Heterogeneity1"] +
    0.3 * data_cleaned2["Intensity Heterogeneity3"] +
    0.2 * data_cleaned2["Intensity Heterogeneity4"]
)
data_cleaned2["Composite Metric Layered"] = composite_metric_layered


#Scatter plot for "Structure Heterogeneity" vs "Whole Strength Mean"
plt.figure(figsize=(13, 13))
plt.scatter(
    data_cleaned2["Structure Heterogeneity"],
    # data_cleaned2["Intensity Heterogeneity1"],
    data_cleaned2["Composite Metric Layered"],
    alpha=0.5,
    label="Structure vs Composite Metric Layered"
)
plt.xlabel("Structure Heterogeneity")
plt.ylabel("Composite Metric Layered")
plt.title("Scatter Plot: Structure vs Composite Metric Layered")
plt.legend()
plt.grid(True)
plt.show()

# Correlation heatmap for all metrics
plt.figure(figsize=(15, 15))
corr_matrix = data_cleaned2.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap: Structure vs Intensity Metrics")
plt.show()


# Re-import necessary libraries and reload the dataset
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
#
# # Reload dataset
# data_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128.pkl'
# data_dict = pickle.load(open(data_path, "rb"))
#
# # Extract relevant metrics
# structure_heterogeneity = [
#     data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1)
#     for data in data_dict.values()
# ]
#
# data_cleaned = pd.DataFrame({
#     "Structure Heterogeneity": structure_heterogeneity,
#     "Whole Strength Mean": [data["whole_strength_mean"] for data in data_dict.values()],
#     "Non-Airway Strength Mean": [data["non_airway_strength_mean"] for data in data_dict.values()],
#     "Airway Strength Mean": [data["airway_strength_mean"] for data in data_dict.values()],
#     "Boundary-Inner Contrast": [data["boundary_inner_contrast"] for data in data_dict.values()],
#     "Airway/Non-Airway Mean Ratio": [data["airway_non_airway_mean_ratio"] for data in data_dict.values()],
#     "Whole Strength Variance": [data["whole_strength_var"] for data in data_dict.values()],
# }).replace([np.inf, -np.inf], np.nan).dropna()
#
# # Recalculate voxel intensity heterogeneity metrics
# data_cleaned["Airway Strength Variance / Mean"] = data_cleaned["Airway Strength Mean"].apply(
#     lambda x: np.nan if x == 0 else data_cleaned["Whole Strength Variance"] / x)
# data_cleaned["Boundary-Inner Contrast"] = data_cleaned["Boundary-Inner Contrast"]
# data_cleaned["Airway/Non-Airway Mean Ratio"] = data_cleaned["Airway/Non-Airway Mean Ratio"]
# data_cleaned["Whole Strength Variance / Mean"] = data_cleaned["Whole Strength Variance"] / data_cleaned["Whole Strength Mean"]
# data_cleaned["Non-Airway Strength Variance / Mean"] = data_cleaned["Non-Airway Strength Mean"].apply(
#     lambda x: np.nan if x == 0 else data_cleaned["Whole Strength Variance"] / x)
#
# # Metrics summary
# metric_summary = pd.DataFrame({
#     "Metric Name": [
#         "Airway Strength Variance / Mean",
#         "Boundary-Inner Contrast",
#         "Airway/Non-Airway Mean Ratio",
#         "Whole Strength Variance / Mean",
#         "Non-Airway Strength Variance / Mean"
#     ],
#     "Formula": [
#         "Airway Strength Variance / Airway Strength Mean",
#         "Boundary Strength Mean - Inner Strength Mean",
#         "Airway Strength Mean / Non-Airway Strength Mean",
#         "Whole Strength Variance / Whole Strength Mean",
#         "Non-Airway Strength Variance / Non-Airway Strength Mean"
#     ],
#     "Example Value": [
#         data_cleaned["Airway Strength Variance / Mean"].mean(),
#         data_cleaned["Boundary-Inner Contrast"].mean(),
#         data_cleaned["Airway/Non-Airway Mean Ratio"].mean(),
#         data_cleaned["Whole Strength Variance / Mean"].mean(),
#         data_cleaned["Non-Airway Strength Variance / Mean"].mean()
#     ]
# })
#
# # Scatter plots: Structural heterogeneity vs intensity metrics
# metrics = [
#     "Airway Strength Variance / Mean",
#     "Boundary-Inner Contrast",
#     "Airway/Non-Airway Mean Ratio",
#     "Whole Strength Variance / Mean",
#     "Non-Airway Strength Variance / Mean"
# ]
#
# fig, axs = plt.subplots(3, 2, figsize=(15, 18))
# axs = axs.flatten()
# for i, metric in enumerate(metrics):
#     axs[i].scatter(data_cleaned["Structure Heterogeneity"], data_cleaned[metric], alpha=0.5, label=metric)
#     axs[i].set_xlabel("Structure Heterogeneity")
#     axs[i].set_ylabel(metric)
#     axs[i].set_title(f"Scatter Plot: Structure vs {metric}")
#     axs[i].grid(True)
# axs[-1].axis('off')  # Remove unused subplot space
# plt.tight_layout()
# plt.show()
#
# # Correlation heatmap for all metrics
# plt.figure(figsize=(12, 10))
# correlation_matrix = data_cleaned.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
# plt.title("Correlation Heatmap: Intensity and Structure Heterogeneity Metrics")
# plt.show()


# Perform PCA and linear regression to evaluate low correlation metrics' independence

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Prepare data for PCA and regression
features = data_cleaned2[[
    "Intensity Heterogeneity1",
    "Intensity Heterogeneity3",
    "Intensity Heterogeneity4",
    "Composite Metric Layered"
]]
target = data_cleaned2["Structure Heterogeneity"]

# Step 1: Perform PCA on intensity metrics to evaluate their independence
pca = PCA(n_components=2)  # Reduce to 2 components for simplicity
pca_result = pca.fit_transform(features)

# Explained variance ratio (PCA components importance)
explained_variance = pca.explained_variance_ratio_

# Step 2: Perform linear regression to test each metric's relationship with Structure Heterogeneity
regression_results = {}
for col in features.columns:
    model = LinearRegression()
    model.fit(features[[col]], target)
    predictions = model.predict(features[[col]])
    r2 = r2_score(target, predictions)
    regression_results[col] = r2

# Display PCA results and regression scores
pca_summary = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(explained_variance))],
    "Explained Variance Ratio": explained_variance
})

regression_summary = pd.DataFrame({
    "Metric": regression_results.keys(),
    "R^2 Score (with Structure Heterogeneity)": regression_results.values()
})

#可视化dataframe regression_summary
print(regression_summary)
print(pca_summary)
# import ace_tools as tools; tools.display_dataframe_to_user(name="PCA Results Summary", dataframe=pca_summary)
# tools.display_dataframe_to_user(name="Linear Regression Results", dataframe=regression_summary)



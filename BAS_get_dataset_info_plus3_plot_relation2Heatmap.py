import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data dictionary
data_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128.pkl'
data_dict = pickle.load(open(data_path, "rb"))

# Extract key statistics for heterogeneity analysis
structure_heterogeneity = [data["airway_pixel_num_boundary"] / data["airway_pixel_num_inner"]
                           if data["airway_pixel_num_inner"] > 0 else np.nan
                            for data in data_dict.values()]
# intensity_heterogeneity = [data["airway_strength_var"] / max(data["airway_strength_mean"], 1)
#                            for data in data_dict.values()]
#
#
# # Separate data into two groups: with airway and without airway
# structure_heterogeneity_with_airway = []
# intensity_heterogeneity_with_airway = []
# structure_heterogeneity_all = []
# intensity_heterogeneity_all = []
#
# for data in data_dict.values():
#     # Check if there are airways in the data
#     if data["airway_pixel_num"] > 0:
#         # For data with airway, calculate heterogeneity indicators and add to "with_airway" lists
#         structure_heterogeneity_with_airway.append(
#             data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1))
#         intensity_heterogeneity_with_airway.append(data["airway_strength_var"] / max(data["airway_strength_mean"], 1))
#
#     # Calculate for all data, regardless of airway presence
#     structure_heterogeneity_all.append(data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1))
#     intensity_heterogeneity_all.append(data["airway_strength_var"] / max(data["airway_strength_mean"], 1))

# Intensity_heterogeneity_1 = [
#     data["boundary_strength_mean"] / max(data["inner_strength_mean"], 1)
#     for data in data_dict.values()
# ]
# Intensity_heterogeneity_2 = [
#     data["airway_strength_mean"] / max(data["non_airway_strength_mean"], 1)
#     for data in data_dict.values()
# ]
#
# Intensity_heterogeneity_3 = [
#     data["airway_strength_var"] / max(data["whole_strength_var"], 1)
#     for data in data_dict.values()
# ]
#
# Intensity_heterogeneity_4 = [
#     data["airway_strength_mean"] / max(data["whole_strength_mean"], 1)
#     for data in data_dict.values()
# ]
#
# Intensity_heterogeneity_5 = [
#     data["airway_strength_var"] / max(data["whole_strength_mean"], 1)
#     for data in data_dict.values()
# ]
# data_cleaned2 = pd.DataFrame({
#     "Structure Heterogeneity": structure_heterogeneity,
#     "Intensity Heterogeneity1": Intensity_heterogeneity_1,
#     "Intensity Heterogeneity2": Intensity_heterogeneity_2,
#     "Intensity Heterogeneity3": Intensity_heterogeneity_3,
#     "Intensity Heterogeneity4": Intensity_heterogeneity_4,
#     "Intensity Heterogeneity5": Intensity_heterogeneity_5,
# }).replace([np.inf, -np.inf], np.nan).dropna()
# composite_metric_layered = (
#     0.5 * data_cleaned2["Intensity Heterogeneity1"] +
#     0.3 * data_cleaned2["Intensity Heterogeneity3"] +
#     0.2 * data_cleaned2["Intensity Heterogeneity4"]
# )
# data_cleaned2["Composite Metric Layered"] = composite_metric_layered
# intensity_heterogeneity=data_cleaned2["Composite Metric Layered"]

# Initialize an empty list to store the composite metric
intensity_heterogeneity = []
# Loop through each row in the data dictionary to calculate metrics
for _, data in data_dict.items():
    # Extract necessary variables for each row
    if data["airway_pixel_num"] > 0:
        boundary_strength_mean = data.get("boundary_strength_mean", np.nan)
        inner_strength_mean = data.get("inner_strength_mean", np.nan)  # Allow actual value, avoid hardcoding 1
        airway_strength_mean = data.get("airway_strength_mean", np.nan)
        non_airway_strength_mean = data.get("non_airway_strength_mean", np.nan)
        airway_strength_var = data.get("airway_strength_var", np.nan)
        whole_strength_mean = data.get("whole_strength_mean", np.nan)
        whole_strength_var = data.get("whole_strength_var", np.nan)

        # Compute each heterogeneity metric using its formula
        # Handle division by zero by checking if the denominator is > 0
        if inner_strength_mean != 0:
            intensity_heterogeneity1 = boundary_strength_mean / inner_strength_mean
        else:
            intensity_heterogeneity1 = np.nan

        if whole_strength_var != 0:
            intensity_heterogeneity3 = airway_strength_var / whole_strength_var
        else:
            intensity_heterogeneity3 = np.nan

        if whole_strength_mean != 0:
            intensity_heterogeneity4 = airway_strength_mean / whole_strength_mean
        else:
            intensity_heterogeneity4 = np.nan

        # Accumulate squared values for valid metrics
        metric_sum = 0
        count = 0
        for metric in [intensity_heterogeneity1, intensity_heterogeneity3, intensity_heterogeneity4]:
            if np.isfinite(metric):  # Ensure the metric is valid
                metric_sum += metric ** 2
                count += 1
        # Calculate the composite metric for this row
        composite_value = np.sqrt(metric_sum / count) if count > 0 else np.nan
        # composite_value = 0.3*intensity_heterogeneity1 + 0.4*intensity_heterogeneity3 + 0.3*intensity_heterogeneity4
        intensity_heterogeneity.append(composite_value)



# Filter for finite values for scatter and heatmap
filtered_data = [(s, i) for s, i in zip(structure_heterogeneity, intensity_heterogeneity) if np.isfinite(s) and np.isfinite(i)]
structure_heterogeneity_filtered, intensity_heterogeneity_filtered = zip(*filtered_data)

# Plot all four graphs on a single figure for comparison
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# # Histogram of structure heterogeneity for data with airway
# axs[0, 0].hist(structure_heterogeneity_with_airway, bins=30, alpha=0.7, edgecolor='black')
# axs[0, 0].set_xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
# axs[0, 0].set_ylabel("Frequency")
# axs[0, 0].set_title("Distribution of Structure Heterogeneity (With Airway)")
# axs[0, 0].grid(True)
#
# # Histogram of intensity heterogeneity for data with airway
# axs[0, 1].hist(intensity_heterogeneity_with_airway, bins=30, alpha=0.7, edgecolor='black')
# axs[0, 1].set_xlabel("Intensity Heterogeneity (Variance/Mean)")
# axs[0, 1].set_ylabel("Frequency")
# axs[0, 1].set_title("Distribution of Intensity Heterogeneity (With Airway)")
# axs[0, 1].grid(True)



# Scatter plot of Structure vs. Intensity Heterogeneity
axs[1, 0].scatter(structure_heterogeneity_filtered, intensity_heterogeneity_filtered, alpha=0.5, color='purple')
axs[1, 0].set_xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
axs[1, 0].set_ylabel("Intensity Heterogeneity (Variance/Mean)")
axs[1, 0].set_title("Scatter Plot of Structure vs Intensity Heterogeneity")
axs[1, 0].grid(True)
correlation_coefficient = np.corrcoef(structure_heterogeneity_filtered, intensity_heterogeneity_filtered)[0, 1]
axs[1, 0].text(0.05, 0.95, f"Correlation: {correlation_coefficient:.2f}", transform=axs[1, 0].transAxes, fontsize=12, verticalalignment='top')

# 2D Heatmap (Hexbin plot) for density analysis
hb = axs[1, 1].hexbin(structure_heterogeneity_filtered, intensity_heterogeneity_filtered, gridsize=30, cmap='coolwarm', mincnt=1)
axs[1, 1].set_xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
axs[1, 1].set_ylabel("Intensity Heterogeneity (Variance/Mean)")
axs[1, 1].set_title("Density Heatmap of Structure and Intensity Heterogeneity")
axs[1, 1].grid(True)
fig.colorbar(hb, ax=axs[1, 0], label='Density')

plt.tight_layout()
plt.show()
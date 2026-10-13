import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the data dictionary
data_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128.pkl'
data_dict = pickle.load(open(data_path, "rb"))

# Extract key statistics for heterogeneity analysis
structure_heterogeneity = [data["airway_pixel_num_boundary"] / data["airway_pixel_num_inner"]
                           if data["airway_pixel_num_inner"] > 0 else np.nan
                            for data in data_dict.values()]
intensity_heterogeneity5 = [-abs(data["airway_strength_var"] / data["airway_strength_mean"])
                            if data["airway_strength_mean"] != 0 else np.nan
                            for data in data_dict.values()]
intensity_heterogeneity = [-abs(data["boundary_strength_mean"] / data["inner_strength_mean"])
                            if data["inner_strength_mean"] != 0 else np.nan
                            for data in data_dict.values()]
intensity_heterogeneity2 = [-abs(data["airway_strength_mean"] / data["non_airway_strength_mean"])
                            if data["non_airway_strength_mean"] != 0 else np.nan
                            for data in data_dict.values()]
intensity_heterogeneity3 = [-abs(data["airway_strength_var"] / data["whole_strength_var"])
                            if data["whole_strength_var"] != 0 else np.nan
                            for data in data_dict.values()]
intensity_heterogeneity4 = [-abs(data["airway_strength_mean"] / data["whole_strength_mean"])
                            if data["whole_strength_mean"] != 0 else np.nan
                            for data in data_dict.values()]

# Separate data into two groups: with airway and without airway
structure_heterogeneity_with_airway = []
intensity_heterogeneity_with_airway = []

for data in data_dict.values():
    # Check if there are airways in the data
    if data["airway_pixel_num"] > 0:
        # For data with airway, calculate heterogeneity indicators and add to "with_airway" lists
        structure_heterogeneity_with_airway.append(
            data["airway_pixel_num_boundary"] / data["airway_pixel_num_inner"] if data["airway_pixel_num_inner"] > 0 else np.nan )
        intensity_heterogeneity_with_airway.append(-abs(data["boundary_strength_mean"] / data["inner_strength_mean"])
                            if data["inner_strength_mean"] != 0 else np.nan
                                                   )
plt.figure()
plt.hist(structure_heterogeneity_with_airway, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
plt.ylabel("Frequency")
plt.title("Distribution of Structure Heterogeneity (With Airway)")
plt.grid(True)

plt.tight_layout()
plt.show()

# Histogram of intensity heterogeneity for data with airway
plt.figure()
plt.hist(intensity_heterogeneity_with_airway, bins=30, alpha=0.7, edgecolor='black')
plt.xlabel("Intensity Heterogeneity (Variance/Mean)")
plt.ylabel("Frequency")
plt.title("Distribution of Intensity Heterogeneity (With Airway)")
plt.grid(True)
plt.show()

# Ensure both arrays have the same length by filtering non-finite values for both simultaneously
filtered_data = [(s, i) for s, i in zip(structure_heterogeneity, intensity_heterogeneity) if np.isfinite(s) and np.isfinite(i)]
structure_heterogeneity_filtered, intensity_heterogeneity_filtered = zip(*filtered_data)

# Scatter plot of Structure vs. Intensity Heterogeneity
plt.figure(figsize=(8, 6))
plt.scatter(structure_heterogeneity_filtered, intensity_heterogeneity_filtered, alpha=0.5, color='purple')
plt.xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
plt.ylabel("Intensity Heterogeneity (Variance/Mean)")
plt.title("Scatter Plot of Structure vs Intensity Heterogeneity")
plt.grid(True)

# Calculate and display correlation coefficient for reference
correlation_coefficient = np.corrcoef(structure_heterogeneity_filtered, intensity_heterogeneity_filtered)[0, 1]
plt.figtext(0.15, 0.85, f"Correlation Coefficient: {correlation_coefficient:.2f}", fontsize=12)

plt.show()

# 2D Heatmap (Hexbin plot) for density analysis of Structure vs Intensity Heterogeneity
plt.figure(figsize=(8, 6))
plt.hexbin(structure_heterogeneity_filtered, intensity_heterogeneity_filtered, gridsize=30, cmap='coolwarm', mincnt=1)
plt.colorbar(label='Density')
plt.xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
plt.ylabel("Intensity Heterogeneity (Variance/Mean)")
plt.title("Density Heatmap of Structure and Intensity Heterogeneity")
plt.grid(True)

plt.show()

# # Visualization: Scatter plot and histograms of the heterogeneity indicators
# plt.figure(figsize=(12, 5))
#
# # Scatter plot of structure vs intensity heterogeneity
# plt.subplot(1, 2, 1)
# plt.scatter(structure_heterogeneity, intensity_heterogeneity, alpha=0.5)
# plt.xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
# plt.ylabel("Intensity Heterogeneity (Variance/Mean)")
# plt.title("Structure vs Intensity Heterogeneity")
# plt.grid(True)
#
# # Histogram of structure heterogeneity
# plt.subplot(1, 2, 2)
# plt.hist(structure_heterogeneity, bins=30, alpha=0.7, edgecolor='black')
# plt.xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
# plt.ylabel("Frequency")
# plt.title("Distribution of Structure Heterogeneity")
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()
#
# # Histogram of intensity heterogeneity in a separate figure
# plt.figure()
# plt.hist(intensity_heterogeneity, bins=30, alpha=0.7, edgecolor='black')
# plt.xlabel("Intensity Heterogeneity (Variance/Mean)")
# plt.ylabel("Frequency")
# plt.title("Distribution of Intensity Heterogeneity")
# plt.grid(True)
#
# plt.show()

# # Visualization for all data (with and without airway)
# plt.figure(figsize=(12, 5))
# plt.suptitle("Heterogeneity Analysis (All Data)")

# # Scatter plot of structure vs intensity heterogeneity for all data
# plt.subplot(1, 2, 1)
# plt.scatter(structure_heterogeneity_all, intensity_heterogeneity_all, alpha=0.5)
# plt.xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
# plt.ylabel("Intensity Heterogeneity (Variance/Mean)")
# plt.title("Structure vs Intensity Heterogeneity (All Data)")
# plt.grid(True)

# # Histogram of structure heterogeneity for all data
# plt.subplot(1, 2, 2)
# plt.hist(structure_heterogeneity_all, bins=30, alpha=0.7, edgecolor='black')
# plt.xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
# plt.ylabel("Frequency")
# plt.title("Distribution of Structure Heterogeneity (All Data)")
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Histogram of intensity heterogeneity for all data
# plt.figure()
# plt.hist(intensity_heterogeneity_all, bins=30, alpha=0.7, edgecolor='black')
# plt.xlabel("Intensity Heterogeneity (Variance/Mean)")
# plt.ylabel("Frequency")
# plt.title("Distribution of Intensity Heterogeneity (All Data)")
# plt.grid(True)
# plt.show()


    # # Calculate for all data, regardless of airway presence
    # structure_heterogeneity_all.append(data["airway_pixel_num_boundary"] / data["airway_pixel_num_inner"] if data["airway_pixel_num_inner"] > 0 else np.nan)
    # intensity_heterogeneity_all.append(-data["airway_strength_var"] / data["airway_strength_mean"] if data["airway_pixel_num_inner"] > 0 else np.nan)

# # Visualization for data with airway
# plt.figure(figsize=(12, 5))
# plt.suptitle("Heterogeneity Analysis (Only Data with Airway)")

# # Scatter plot of structure vs intensity heterogeneity for data with airway
# plt.subplot(1, 2, 1)
# plt.scatter(structure_heterogeneity_with_airway, intensity_heterogeneity_with_airway, alpha=0.5)
# plt.xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
# plt.ylabel("Intensity Heterogeneity (Variance/Mean)")
# plt.title("Structure vs Intensity Heterogeneity (With Airway)")
# plt.grid(True)

# # Histogram of structure heterogeneity for data with airway
# plt.subplot(1, 2, 2)
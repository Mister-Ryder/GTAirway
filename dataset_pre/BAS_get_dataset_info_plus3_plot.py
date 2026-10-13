import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the data dictionary
data_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128_origin.pkl'
data_dict = pickle.load(open(data_path, "rb"))

# Extract key statistics for heterogeneity analysis
structure_heterogeneity = [data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1)
                           for data in data_dict.values()]
intensity_heterogeneity = [data["airway_strength_var"] / max(data["airway_strength_mean"], 1)
                           for data in data_dict.values()]


# Separate data into two groups: with airway and without airway
structure_heterogeneity_with_airway = []
intensity_heterogeneity_with_airway = []
structure_heterogeneity_all = []
intensity_heterogeneity_all = []

for data in data_dict.values():
    # Check if there are airways in the data
    if data["airway_pixel_num"] > 0:
        # For data with airway, calculate heterogeneity indicators and add to "with_airway" lists
        structure_heterogeneity_with_airway.append(
            data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1))
        intensity_heterogeneity_with_airway.append(data["airway_strength_var"] / max(data["airway_strength_mean"], 1))

    # Calculate for all data, regardless of airway presence
    structure_heterogeneity_all.append(data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1))
    intensity_heterogeneity_all.append(data["airway_strength_var"] / max(data["airway_strength_mean"], 1))

# Filter for finite values for scatter and heatmap
filtered_data = [(s, i) for s, i in zip(structure_heterogeneity, intensity_heterogeneity) if np.isfinite(s) and np.isfinite(i)]
structure_heterogeneity_filtered, intensity_heterogeneity_filtered = zip(*filtered_data)

# Plot all four graphs on a single figure for comparison
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Histogram of structure heterogeneity for data with airway
axs[0, 0].hist(structure_heterogeneity_with_airway, bins=30, alpha=0.7, edgecolor='black')
axs[0, 0].set_xlabel("Structure Heterogeneity (Boundary/Inner Ratio)")
axs[0, 0].set_ylabel("Frequency")
axs[0, 0].set_title("Distribution of Structure Heterogeneity (With Airway)")
axs[0, 0].grid(True)

# Histogram of intensity heterogeneity for data with airway
axs[0, 1].hist(intensity_heterogeneity_with_airway, bins=30, alpha=0.7, edgecolor='black')
axs[0, 1].set_xlabel("Intensity Heterogeneity (Variance/Mean)")
axs[0, 1].set_ylabel("Frequency")
axs[0, 1].set_title("Distribution of Intensity Heterogeneity (With Airway)")
axs[0, 1].grid(True)



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
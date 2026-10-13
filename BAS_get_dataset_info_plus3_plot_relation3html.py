#V1

# # Recalculate all variables and combine into a single cell for clarity and correct execution
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import pickle
#
# # Load the provided pkl data
# pkl_file_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_positions_and_ids.pkl'
# with open(pkl_file_path, 'rb') as f:
#     data_dict = pickle.load(f)
#
# # Define the target image ID for visualization
# target_image_id = '0140'  # Using the provided ID example
#
# # Extract relevant data for the target image ID
# positions = []
# structure_heterogeneity = []
# intensity_heterogeneity = []
#
# for data in data_dict.values():
#     if data["image_id"] == target_image_id:
#         positions.append(data["position"])
#         structure_heterogeneity.append(data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1))
#         intensity_heterogeneity.append(data["airway_strength_var"] / max(data["airway_strength_mean"], 1))
#
# # Separate positions into x, y, z coordinates
# x = [pos[0]+32 for pos in positions]
# y = [pos[1]+32 for pos in positions]
# z = [pos[2]+32 for pos in positions]
#
# # Normalize intensity heterogeneity for radius scaling
# max_intensity = max(intensity_heterogeneity) if intensity_heterogeneity else 1
# radii = [i / max_intensity * 30 for i in intensity_heterogeneity]  # Scale radii for visibility
#
# # Create 3D plot with spheres
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot each position as a sphere based on intensity (radius) and structure heterogeneity (color)
# for xi, yi, zi, r, sh in zip(x, y, z, radii, structure_heterogeneity):
#     u = np.linspace(0, 2 * np.pi, 30)
#     v = np.linspace(0, np.pi, 15)
#     x_sphere = r * np.outer(np.cos(u), np.sin(v)) + xi
#     y_sphere = r * np.outer(np.sin(u), np.sin(v)) + yi
#     z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v)) + zi
#     ax.plot_surface(x_sphere, y_sphere, z_sphere, color=plt.cm.viridis(sh / max(structure_heterogeneity)), alpha=0.8)
#
# # Set up color bar for structure heterogeneity
# mappable = plt.cm.ScalarMappable(cmap='viridis')
# mappable.set_array(structure_heterogeneity)
# cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
# cbar.set_label('Structure Heterogeneity')
#
# # Label axes
# ax.set_xlabel('X Position')
# ax.set_ylabel('Y Position')
# ax.set_zlabel('Z Position')
# plt.title(f"Heterogeneity Visualization for Image ID: {target_image_id} with Sphere Radius")
#
# # Display the plot
# plt.show()



#V2

# import pickle
# import numpy as np
# import plotly.graph_objects as go
# import re
#
# # Step 1: Load data from the provided pkl file
# pkl_file_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_positions_and_ids.pkl'
# with open(pkl_file_path, 'rb') as f:
#     data_dict = pickle.load(f)
#
# # Step 2: Load image size info from the info file
# def parse_info_file(file_path):
#     image_info = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             match = re.search(r"Image Path: .*/(\d+).nii.gz, Image Size: \((\d+), (\d+), (\d+)\)", line)
#             if match:
#                 img_id = match.group(1)
#                 x_size, y_size, z_size = int(match.group(2)), int(match.group(3)), int(match.group(4))
#                 image_info[img_id] = (x_size, y_size, z_size)
#     return image_info
#
# info_file_path = 'dataset_info/info_BAS.txt'  # Replace with your info file path
# image_info = parse_info_file(info_file_path)
#
# # Step 3: Define the target image ID and extract data for visualization
# target_image_id = '0292'  # Replace with the desired image ID
# positions = []
# structure_heterogeneity = []
# intensity_heterogeneity = []
#
# for data in data_dict.values():
#     if data["image_id"] == target_image_id:
#         positions.append(data["position"])
#         structure_heterogeneity.append(data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1))
#         intensity_heterogeneity.append(data["airway_strength_var"] / max(data["airway_strength_mean"], 1))
#
# # Separate position data into x, y, z coordinates
# x = [pos[0]+32 for pos in positions]
# y = [pos[1]+32 for pos in positions]
# z = [pos[2]+32 for pos in positions]
#
# # Step 4: Filter out points with invalid heterogeneity values (e.g., NaN or zero)
# valid_indices = [
#     i for i, (sh, ih) in enumerate(zip(structure_heterogeneity, intensity_heterogeneity))
#     if sh > 0 and np.isfinite(sh) and ih > 0 and np.isfinite(ih)
# ]
# filtered_x = [x[i] for i in valid_indices]
# filtered_y = [y[i] for i in valid_indices]
# filtered_z = [z[i] for i in valid_indices]
# filtered_radii = [structure_heterogeneity[i] / max(structure_heterogeneity) * 30+5 for i in valid_indices]
# filtered_colors = [intensity_heterogeneity[i] for i in valid_indices]
#
# # Step 5: Get image size for setting the axes range
# image_size_x, image_size_y, image_size_z = image_info.get(target_image_id, (1, 1, 1))
#
# # Step 6: Create the interactive 3D plot with Plotly
# fig = go.Figure(data=[
#     go.Scatter3d(
#         x=filtered_x,
#         y=filtered_y,
#         z=filtered_z,
#         mode='markers',
#         marker=dict(
#             size=filtered_radii,
#             color=filtered_colors,
#             colorscale='Viridis',
#             colorbar=dict(title="Intensity Heterogeneity"),
#             opacity=0.8
#         ),
#         text=[f"Structure Heterogeneity: {sh:.2f}, Intensity Heterogeneity: {ih:.2f}"
#               for i, (sh, ih) in enumerate(zip(structure_heterogeneity, intensity_heterogeneity)) if i in valid_indices]
#     )
# ])
#
# # Step 7: Configure axes to match image dimensions
# fig.update_layout(
#     scene=dict(
#         xaxis=dict(title='X Position', range=[0, image_size_x]),
#         yaxis=dict(title='Y Position', range=[0, image_size_y]),
#         zaxis=dict(title='Z Position', range=[0, image_size_z])
#     ),
#     title=f'Heterogeneity Visualization for Image ID: {target_image_id} (Image Size Axes)',
#     margin=dict(l=0, r=0, b=0, t=40)
# )
#
# # Step 8: Save the plot as an HTML file
# output_html_path = "heterogeneity_visualization_interactive.html"  # Specify output file path
# fig.write_html(output_html_path)
# print(f"Interactive plot saved to {output_html_path}")


# V3


# import pickle
# import numpy as np
# import plotly.graph_objects as go
# import re
# import os
#
# # Step 1: Load data from the provided pkl file
# pkl_file_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_positions_and_ids.pkl'
# with open(pkl_file_path, 'rb') as f:
#     data_dict = pickle.load(f)
#
#
# # Step 2: Load image size info from the info file
# def parse_info_file(file_path):
#     image_info = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             match = re.search(r"Image Path: .*/(\d+).nii.gz, Image Size: \((\d+), (\d+), (\d+)\)", line)
#             if match:
#                 img_id = match.group(1)
#                 x_size, y_size, z_size = int(match.group(2)), int(match.group(3)), int(match.group(4))
#                 image_info[img_id] = (x_size, y_size, z_size)
#     return image_info
#
#
# info_file_path = 'dataset_info/info_BAS.txt'  # Replace with your info file path
# image_info = parse_info_file(info_file_path)
#
# # Step 3: Get all unique image IDs in the pkl data
# unique_image_ids = set(data["image_id"] for data in data_dict.values())
#
# # Output directory
# output_dir = "visualization_outputs"
# os.makedirs(output_dir, exist_ok=True)
#
# # Step 4: Loop through each unique image_id
# for target_image_id in unique_image_ids:
#     # Collect data for the current image_id
#     positions = []
#     structure_heterogeneity = []
#     intensity_heterogeneity = []
#
#     for data in data_dict.values():
#         if data["image_id"] == target_image_id:
#             positions.append(data["position"])
#             structure_heterogeneity.append(data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1))
#             intensity_heterogeneity.append(data["airway_strength_var"] / max(data["airway_strength_mean"], 1))
#
#     # Separate position data into x, y, z coordinates
#     x = [pos[0] + 32 for pos in positions]
#     y = [pos[1] + 32 for pos in positions]
#     z = [pos[2] + 32 for pos in positions]
#
#     # Filter out invalid values
#     valid_indices = [
#         i for i, (sh, ih) in enumerate(zip(structure_heterogeneity, intensity_heterogeneity))
#         if sh > 0 and np.isfinite(sh) and ih > 0 and np.isfinite(ih)
#     ]
#     filtered_x = [x[i] for i in valid_indices]
#     filtered_y = [y[i] for i in valid_indices]
#     filtered_z = [z[i] for i in valid_indices]
#     filtered_radii = [structure_heterogeneity[i] / max(structure_heterogeneity) * 30 + 5 for i in valid_indices]
#     filtered_colors = [intensity_heterogeneity[i] for i in valid_indices]
#
#     # Get image size for setting the axes range
#     image_size_x, image_size_y, image_size_z = image_info.get(target_image_id, (1, 1, 1))
#
#     # Step 5: Create the interactive 3D plot with Plotly
#     fig = go.Figure(data=[
#         go.Scatter3d(
#             x=filtered_x,
#             y=filtered_y,
#             z=filtered_z,
#             mode='markers',
#             marker=dict(
#                 size=filtered_radii,
#                 color=filtered_colors,
#                 colorscale='Viridis',
#                 colorbar=dict(title="Intensity Heterogeneity"),
#                 opacity=0.8
#             ),
#             text=[f"Structure Heterogeneity: {sh:.2f}, Intensity Heterogeneity: {ih:.2f}"
#                   for i, (sh, ih) in enumerate(zip(structure_heterogeneity, intensity_heterogeneity)) if
#                   i in valid_indices]
#         )
#     ])
#
#     # Configure axes to match image dimensions
#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(title='X Position', range=[0, image_size_x]),
#             yaxis=dict(title='Y Position', range=[0, image_size_y]),
#             zaxis=dict(title='Z Position', range=[0, image_size_z])
#         ),
#         title=f'Heterogeneity Visualization for Image ID: {target_image_id} (Image Size Axes)',
#         margin=dict(l=0, r=0, b=0, t=40)
#     )
#
#     # Step 6: Save the plot as an HTML file
#     html_path = os.path.join(output_dir, f"heterogeneity_visualization_{target_image_id}.html")
#     fig.write_html(html_path)
#
#     # Step 7: Save the plot as a static JPG image
#     jpg_path = os.path.join(output_dir, f"heterogeneity_visualization_{target_image_id}.jpg")
#     fig.write_image(jpg_path, width=800, height=600)  # Specify resolution for the JPG output
#
#     print(f"Saved {target_image_id} visualization to {html_path} and {jpg_path}")


#V4

import pickle
import numpy as np
import plotly.graph_objects as go
import re
import os

# Step 1: Load data from the provided pkl file
pkl_file_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_positions_and_ids.pkl'
with open(pkl_file_path, 'rb') as f:
    data_dict = pickle.load(f)


# Step 2: Load image size info from the info file
def parse_info_file(file_path):
    image_info = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r"Image Path: .*/(\d+).nii.gz, Image Size: \((\d+), (\d+), (\d+)\)", line)
            if match:
                img_id = match.group(1)
                x_size, y_size, z_size = int(match.group(2)), int(match.group(3)), int(match.group(4))
                image_info[img_id] = (x_size, y_size, z_size)
    return image_info


info_file_path = 'dataset_info/info_BAS.txt'
image_info = parse_info_file(info_file_path)

# Step 3: Get all unique image IDs and global max/min heterogeneity values
unique_image_ids = set(data["image_id"] for data in data_dict.values())
all_structure_values = []
all_intensity_values = []

for data in data_dict.values():
    sh = data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1)
    ih = data["airway_strength_var"] / max(data["airway_strength_mean"], 1)
    if np.isfinite(sh) and sh > 0:
        all_structure_values.append(sh)
    if np.isfinite(ih) and ih > 0:
        all_intensity_values.append(ih)

# Calculate global min and max values for heterogeneity
global_min_structure = min(all_structure_values)
global_max_structure = max(all_structure_values)
global_min_intensity = min(all_intensity_values)
global_max_intensity = max(all_intensity_values)

# Output directory
output_dir = "visualization_outputs"
os.makedirs(output_dir, exist_ok=True)

# Step 4: Loop through each unique image_id
for target_image_id in unique_image_ids:
    positions = []
    structure_heterogeneity = []
    intensity_heterogeneity = []

    for data in data_dict.values():
        if data["image_id"] == target_image_id:
            positions.append(data["position"])
            structure_heterogeneity.append(data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1))
            intensity_heterogeneity.append(data["airway_strength_var"] / max(data["airway_strength_mean"], 1))

    # Separate position data into x, y, z coordinates
    x = [pos[0] + 32 for pos in positions]
    y = [pos[1] + 32 for pos in positions]
    z = [pos[2] + 64 for pos in positions]

    # Filter out invalid values
    valid_indices = [
        i for i, (sh, ih) in enumerate(zip(structure_heterogeneity, intensity_heterogeneity))
        if sh > 0 and np.isfinite(sh) and ih > 0 and np.isfinite(ih)
    ]
    filtered_x = [x[i] for i in valid_indices]
    filtered_y = [y[i] for i in valid_indices]
    filtered_z = [z[i] for i in valid_indices]

    # Adjust radii based on global structure heterogeneity range
    filtered_radii = [
        (structure_heterogeneity[i] - global_min_structure) / (global_max_structure - global_min_structure) * 40 + 10
        for i in valid_indices
    ]

    # Adjust color based on global intensity heterogeneity range
    filtered_colors = [
        (intensity_heterogeneity[i] - global_min_intensity) / (global_max_intensity - global_min_intensity)
        for i in valid_indices
    ]

    # Get image size for setting the axes range
    image_size_x, image_size_y, image_size_z = image_info.get(target_image_id, (1, 1, 1))

    # Step 5: Create the interactive 3D plot with Plotly
    fig = go.Figure(data=[
        go.Scatter3d(
            x=filtered_x,
            y=filtered_y,
            z=filtered_z,
            mode='markers',
            marker=dict(
                size=filtered_radii,
                color=filtered_colors,
                colorscale='Viridis',
                cmin=0,  # Set color range to global min
                cmax=1,  # Set color range to global max
                colorbar=dict(title="Intensity Heterogeneity"),
                opacity=0.8
            ),
            text=[f"Structure Heterogeneity: {sh:.2f}, Intensity Heterogeneity: {ih:.2f}"
                  for i, (sh, ih) in enumerate(zip(structure_heterogeneity, intensity_heterogeneity)) if
                  i in valid_indices]
        )
    ])

    # Configure axes to match image dimensions
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X Position', range=[0, image_size_x]),
            yaxis=dict(title='Y Position', range=[0, image_size_y]),
            zaxis=dict(title='Z Position', range=[0, image_size_z])
        ),
        title=f'Heterogeneity Visualization for Image ID: {target_image_id} (Global Scale)',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Step 6: Save the plot as an HTML file
    html_path = os.path.join(output_dir, f"heterogeneity_{target_image_id}.html")
    fig.write_html(html_path)

    # Step 7: Save the plot as a static JPG image
    jpg_path = os.path.join(output_dir, f"heterogeneity_{target_image_id}.jpg")
    fig.write_image(jpg_path, width=800, height=600)  # Specify resolution for the JPG output

    print(f"Saved {target_image_id} visualization to {html_path} and {jpg_path}")

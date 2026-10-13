# import csv
# import numpy as np
# import pickle
# import os
#
# # Load data from the provided pkl file
# pkl_file_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_positions_and_ids.pkl'
# with open(pkl_file_path, 'rb') as f:
#     data_dict = pickle.load(f)
#
# # Define the target image ID to export data for each image
# unique_image_ids = set(data["image_id"] for data in data_dict.values())
#
# # Output directory for CSV files
# output_dir = "heterogeneity_csv_outputs"
# os.makedirs(output_dir, exist_ok=True)
#
# # Loop through each unique image_id
# for target_image_id in unique_image_ids:
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
#     # Filter out invalid values
#     valid_indices = [
#         i for i, (sh, ih) in enumerate(zip(structure_heterogeneity, intensity_heterogeneity))
#         if sh > 0 and np.isfinite(sh) and ih > 0 and np.isfinite(ih)
#     ]
#     filtered_positions = [positions[i] for i in valid_indices]
#     filtered_structure = [structure_heterogeneity[i] for i in valid_indices]
#     filtered_intensity = [intensity_heterogeneity[i] for i in valid_indices]
#
#     # Export each position with its heterogeneity values as a CSV file
#     csv_path = os.path.join(output_dir, f"heterogeneity_{target_image_id}.csv")
#     with open(csv_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["X", "Y", "Z", "StructureHeterogeneity", "IntensityHeterogeneity"])  # Header
#         for pos, sh, ih in zip(filtered_positions, filtered_structure, filtered_intensity):
#             writer.writerow([pos[0], pos[1], pos[2], sh, ih])
#
#     print(f"Exported heterogeneity data for {target_image_id} to {csv_path}")
#
#
# import vtk
#
# # Function to write points to VTK file
# def write_points_to_vtk(positions, structure_heterogeneity, intensity_heterogeneity, file_path, scale_factor=1):
#     # Create a vtkPoints object and add the points
#     points = vtk.vtkPoints()
#     heterogeneity_data = vtk.vtkFloatArray()
#     heterogeneity_data.SetName("StructureHeterogeneity")
#
#     intensity_data = vtk.vtkFloatArray()
#     intensity_data.SetName("IntensityHeterogeneity")
#
#     for pos, sh, ih in zip(positions, structure_heterogeneity, intensity_heterogeneity):
#         # Scale the position to make it more visible in the software
#         scaled_pos = [p * scale_factor for p in pos]
#         points.InsertNextPoint(scaled_pos)
#         heterogeneity_data.InsertNextValue(sh)
#         intensity_data.InsertNextValue(ih)
#
#     # Create a polydata object
#     polydata = vtk.vtkPolyData()
#     polydata.SetPoints(points)
#     polydata.GetPointData().AddArray(heterogeneity_data)
#     polydata.GetPointData().AddArray(intensity_data)
#
#     # Write to file
#     writer = vtk.vtkXMLPolyDataWriter()
#     writer.SetFileName(file_path)
#     writer.SetInputData(polydata)
#     writer.Write()
#
#
# # Save each target image data as vtk file
# for target_image_id in unique_image_ids:
#     # Use the valid positions and heterogeneity values filtered above
#     vtk_path = os.path.join(output_dir, f"heterogeneity_{target_image_id}.vtp")
#     write_points_to_vtk(filtered_positions, filtered_structure, filtered_intensity, vtk_path,scale_factor=10)
#     print(f"Exported heterogeneity data to VTK for {target_image_id} at {vtk_path}")


import numpy as np
import SimpleITK as sitk
import pickle

# 加载CT图像以获取尺寸和空间信息
ct_path = "test_data/0140.nii.gz"  # 替换为您的CT图像路径
ct_image = sitk.ReadImage(ct_path)
ct_shape = ct_image.GetSize()  # 获取图像的尺寸 (X, Y, Z)
ct_spacing = ct_image.GetSpacing()  # 获取图像的空间分辨率
ct_origin = ct_image.GetOrigin()  # 获取图像的原点坐标

# 初始化与CT图像尺寸一致的空白三维数组，用于存储异质性强度值
structure_heterogeneity_image = np.zeros(ct_shape, dtype=np.float32)
intensity_heterogeneity_image = np.zeros(ct_shape, dtype=np.float32)

# 加载异质性分析数据
pkl_file_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_positions_and_ids.pkl'
with open(pkl_file_path, 'rb') as f:
    data_dict = pickle.load(f)

# 目标图像 ID（可以替换为实际值）
target_image_id = '0140'

# 遍历数据字典并填充异质性图像
for data in data_dict.values():
    if data["image_id"] == target_image_id:
        pos = data["position"]
        x, y, z = pos  # 提取三维坐标
        structure_heterogeneity = data["airway_pixel_num_boundary"] / max(data["airway_pixel_num_inner"], 1)
        intensity_heterogeneity = data["airway_strength_var"] / max(data["airway_strength_mean"], 1)

        # 将异质性值填充到对应位置
        structure_heterogeneity_image[x, y, z] = structure_heterogeneity
        intensity_heterogeneity_image[x, y, z] = intensity_heterogeneity

# 将数组转换为SimpleITK图像，设置空间信息与原始CT一致
structure_heterogeneity_sitk = sitk.GetImageFromArray(structure_heterogeneity_image.transpose(2, 1, 0))
structure_heterogeneity_sitk.SetSpacing(ct_spacing)
structure_heterogeneity_sitk.SetOrigin(ct_origin)

intensity_heterogeneity_sitk = sitk.GetImageFromArray(intensity_heterogeneity_image.transpose(2, 1, 0))
intensity_heterogeneity_sitk.SetSpacing(ct_spacing)
intensity_heterogeneity_sitk.SetOrigin(ct_origin)

# 保存为NIfTI文件
sitk.WriteImage(structure_heterogeneity_sitk, "structure_heterogeneity_map.nii.gz")
sitk.WriteImage(intensity_heterogeneity_sitk, "intensity_heterogeneity_map.nii.gz")

print("NIfTI files for heterogeneity maps have been saved.")


# import nibabel as nib
# import numpy as np
# import pickle
# import re
#
# # 加载 .pkl 数据文件
# pkl_file_path = 'dataset_info/3dataset_info_BAS_Skeleton_crops_128_with_positions_and_ids.pkl'
# with open(pkl_file_path, 'rb') as f:
#     data_dict = pickle.load(f)
#
# # 图像维度根据 info_BAS.txt 中的0140信息
# image_size = (512, 512, 588)  # Z, Y, X
#
# # 从 .pkl 文件中提取 `image_id` 为 `0140` 的位置和有效的半径和强度信息
# spheres_info = []
# for data in data_dict.values():
#     if data["image_id"] == "0140":
#         # 提取位置
#         position = data["position"]
#
#         # 计算半径
#         inner_pixels = data["airway_pixel_num_inner"]
#         boundary_pixels = data["airway_pixel_num_boundary"]
#         if inner_pixels > 0 and boundary_pixels > 0:  # 确保非零和正数
#             radius = boundary_pixels / inner_pixels * 10  # 示例缩放因子
#
#             # 计算强度并检查有效性
#             strength_mean = data["airway_strength_mean"]
#             strength_var = data["airway_strength_var"]
#             if strength_mean > 0 and np.isfinite(strength_var):  # 排除 NaN 和零值
#                 intensity_heterogeneity = strength_var / strength_mean
#                 label_value = int(intensity_heterogeneity * 255)
#
#                 # 添加到球体信息列表
#                 spheres_info.append({
#                     'position': (position[2] + 64, position[1] + 32, position[0] + 32),  # 使用偏移修正位置
#                     'radius': radius,
#                     'label': label_value
#                 })
#
# def create_spherical_mask(center, radius, img_shape, label_value):
#     """
#     创建3D球体掩码，给定中心、半径和标签值。
#     """
#     z, y, x = np.ogrid[:img_shape[0], :img_shape[1], :img_shape[2]]
#     distance = np.sqrt((x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2)
#     mask = distance <= radius
#     labeled_sphere = np.zeros(img_shape, dtype=np.int16)
#     labeled_sphere[mask] = label_value
#     return labeled_sphere
#
# # 创建图像体积
# volume = np.zeros(image_size, dtype=np.int16)
#
# # 将球体信息应用到体积数据中
# for sphere in spheres_info:
#     center = sphere['position']
#     radius = sphere['radius']
#     label_value = sphere['label']
#     sphere_mask = create_spherical_mask(center, radius, image_size, label_value)
#     volume = np.maximum(volume, sphere_mask)  # 合并球体，避免标签覆盖
#
# # 调整维度顺序 (Z, Y, X -> X, Y, Z)
# volume_corrected = np.transpose(volume, (2, 1, 0))
#
# # 保存为 NIfTI 文件
# output_path = 'visualization_outputs/heterogeneity_0140_complete_corrected.nii.gz'
# nii_image = nib.Nifti1Image(volume_corrected, affine=np.eye(4))
# nib.save(nii_image, output_path)
#
# print(f"已保存完整的 NIfTI 文件到 {output_path}")
#
#
#
#
# import nibabel as nib
# import numpy as np
#
# # 加载已生成的 NIfTI 文件
# nii_path = 'visualization_outputs/heterogeneity_0140_complete_corrected.nii.gz'
# nii_image = nib.load(nii_path)
# volume = nii_image.get_fdata()
#
# # 调整维度顺序为 (512, 512, 588)，以匹配主图像维度
# volume_corrected = np.transpose(volume, (2, 1, 0))
#
# # 使用相同的仿射矩阵，保存为新的 NIfTI 文件
# corrected_nii_image = nib.Nifti1Image(volume_corrected, affine=nii_image.affine)
# corrected_nii_path = 'visualization_outputs/heterogeneity_0140_corrected.nii.gz'
# nib.save(corrected_nii_image, corrected_nii_path)
#
# print(f"已保存修正后的 NIfTI 文件到 {corrected_nii_path}")




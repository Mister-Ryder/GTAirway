# AirwayMetadataGenerator.py
import os
import numpy as np
import pickle
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm
from skimage.morphology import skeletonize_3d
import skimage.measure as measure


class AirwayMetadataGenerator:
    def __init__(self,
                 raw_data_dir,
                 parsing_dir,
                 output_pkl_path,
                 crop_size=(128, 128, 128),
                 stride=(64, 64, 64)):
        self.raw_data_dir = raw_data_dir
        self.parsing_dir = parsing_dir
        self.output_pkl_path = output_pkl_path
        self.crop_size = crop_size
        self.stride = stride
        self.data_dict = {}

    def _calculate_positions(self, img_shape):
        """计算裁剪块的起始坐标"""
        positions = []
        for z in range(0, img_shape[0], self.stride[0]):
            for y in range(0, img_shape[1], self.stride[1]):
                for x in range(0, img_shape[2], self.stride[2]):
                    z_start = min(z, img_shape[0] - self.crop_size[0])
                    y_start = min(y, img_shape[1] - self.crop_size[1])
                    x_start = min(x, img_shape[2] - self.crop_size[2])
                    positions.append((z_start, y_start, x_start))
        return positions

    def _get_case_ids(self, mode="train"):
        # 获取图像目录中的所有病例ID
        image_dir = os.path.join(self.raw_data_dir, mode, "image")
        case_files = os.listdir(image_dir)
        case_ids = [f.split(".")[0] for f in case_files]
        return case_ids

    def _load_parsing_data(self, case_id, mode="train"):
        """加载解析后的气道树数据"""
        parsing_path = os.path.join(self.parsing_dir, f"{case_id}_parse.nii.gz")
        parent_path = os.path.join(self.parsing_dir, f"{case_id}_parent.npy")
        children_path = os.path.join(self.parsing_dir, f"{case_id}_children.npy")

        # 加载分支解析数据
        try:
            parsing = sitk.GetArrayFromImage(sitk.ReadImage(parsing_path))
        except Exception as e:
            print(f"Error loading parsing data for {case_id}: {e}")
            return None, None

        # 加载代数信息
        generation = None
        if os.path.exists(parent_path) and os.path.exists(children_path):
            try:
                parent_map = np.load(parent_path)
                children_map = np.load(children_path)

                # 计算每个分支的代数
                num_branches = parent_map.shape[0]
                generation = np.zeros(num_branches, dtype=np.uint8)

                # 寻找根节点（气管）
                trachea_index = None
                for i in range(num_branches):
                    if parent_map[i, i] == 1:  # 自己是自己的父节点，通常是气管
                        trachea_index = i
                        break

                if trachea_index is not None:
                    # 气管的代数为0
                    generation[trachea_index] = 0

                    # 使用广度优先搜索计算每个分支的代数
                    queue = [trachea_index]
                    while queue:
                        current = queue.pop(0)
                        current_gen = generation[current]

                        # 找出所有子分支
                        children = np.where(children_map[current, :] > 0)[0]
                        for child in children:
                            generation[child] = current_gen + 1
                            queue.append(child)
                else:
                    print(f"Warning: Could not find trachea for {case_id}")
            except Exception as e:
                print(f"Error processing generation data for {case_id}: {e}")
                generation = None

        return parsing, generation

    # def _compute_anatomical_features(self, img_patch, label_patch, parsing_patch, generation, position, img_shape):
    #     """计算解剖学特征""" V1版本
    #     # 基本指标
    #     airway_pixels = np.sum(label_patch)
    #     if airway_pixels == 0:
    #         return {
    #             'airway_pixel_num': 0,
    #             'generation_type': 0,  # 无气道
    #             'structure_type': 0,  # 无结构
    #             'location_type': 0,  # 未定位
    #             'contains_bifurcation': False,
    #             'airway_density': 0
    #         }
    #
    #     # 从解析数据中提取分支信息
    #     unique_branches = np.unique(parsing_patch[parsing_patch > 0])
    #
    #     # 计算各分支的体积占比
    #     branch_volumes = {}
    #     for branch_id in unique_branches:
    #         if branch_id > 0:  # 确保分支ID有效
    #             branch_volume = np.sum(parsing_patch == branch_id)
    #             branch_volumes[branch_id] = branch_volume / airway_pixels
    #
    #     # 提取主要分支和代数
    #     main_branch = 0
    #     branch_generation = 0
    #     generation_type = 0
    #
    #     if len(branch_volumes) > 0:
    #         main_branch = max(branch_volumes, key=branch_volumes.get)
    #
    #         if generation is not None and main_branch > 0 and main_branch <= len(generation):
    #             branch_generation = generation[main_branch - 1]  # 注意索引偏移
    #
    #             # 生成解剖学分类
    #             # 代数类型: 0=无气道, 1=气管, 2=主支气管, 3=叶支气管, 4=段支气管, 5=细支气管
    #             if branch_generation == 0:
    #                 generation_type = 1  # 气管
    #             elif branch_generation <= 2:
    #                 generation_type = 2  # 主支气管
    #             elif branch_generation <= 4:
    #                 generation_type = 3  # 叶支气管
    #             elif branch_generation <= 6:
    #                 generation_type = 4  # 段支气管
    #             else:
    #                 generation_type = 5  # 细支气管
    #
    #     # 判断是否包含分叉
    #     contains_bifurcation = len(unique_branches) > 1
    #
    #     # 结构类型: 0=无结构, 1=直管道, 2=分叉点, 3=多分支区域
    #     if len(unique_branches) == 0:
    #         structure_type = 0  # 无结构
    #     elif len(unique_branches) == 1:
    #         structure_type = 1  # 直管道
    #     elif contains_bifurcation:
    #         structure_type = 2  # 分叉点
    #     else:
    #         structure_type = 3  # 多分支区域
    #
    #     # 空间位置分类: 基于滑块在完整体积中的相对位置
    #     z_center, y_center, x_center = [p + self.crop_size[i] // 2 for i, p in enumerate(position)]
    #     z_rel = z_center / img_shape[0]
    #
    #     # 上/中/下肺分类 (基于z轴位置)
    #     if z_rel < 0.33:
    #         location_z = 1  # 上肺
    #     elif z_rel < 0.66:
    #         location_z = 2  # 中肺
    #     else:
    #         location_z = 3  # 下肺
    #
    #     # 位置类型: 1=上肺, 2=中肺, 3=下肺, 4=左肺, 5=右肺, 6=气管区域
    #     # 简化版本，只基于z轴位置判断
    #     if generation_type == 1:  # 如果是气管
    #         location_type = 6  # 气管区域
    #     else:
    #         location_type = location_z
    #
    #     airway_density = airway_pixels / (self.crop_size[0] * self.crop_size[1] * self.crop_size[2])
    #
    #     return {
    #         'airway_pixel_num': airway_pixels,
    #         'generation_type': generation_type,
    #         'structure_type': structure_type,
    #         'location_type': location_type,
    #         'contains_bifurcation': contains_bifurcation,
    #         'airway_density': airway_density,
    #         'main_branch_id': main_branch,
    #         'branch_diversity': len(unique_branches)
    #     }

    def _compute_anatomical_features(self, img_patch, label_patch, parsing_patch, generation, position, img_shape):
        """计算解剖学特征"""
        # 基本指标
        airway_pixels = np.sum(label_patch)
        if airway_pixels == 0:
            return {
                'airway_pixel_num': 0,
                'generation_type': 0,  # 无气道
                'structure_type': 0,  # 无结构
                'location_type': 0,  # 未定位
                'contains_bifurcation': False,
                'airway_density': 0
            }

        # 从解析数据中提取分支信息
        unique_branches = np.unique(parsing_patch[parsing_patch > 0])

        # 计算各分支的体积占比
        branch_volumes = {}
        for branch_id in unique_branches:
            if branch_id > 0:  # 确保分支ID有效
                branch_volume = np.sum(parsing_patch == branch_id)
                branch_volumes[branch_id] = branch_volume / airway_pixels

        # 提取主要分支和代数
        main_branch = 0
        branch_generation = 0
        generation_type = 0

        if len(branch_volumes) > 0:
            main_branch = max(branch_volumes, key=branch_volumes.get)

            if generation is not None and main_branch > 0 and main_branch <= len(generation):
                branch_generation = generation[main_branch - 1]  # 注意索引偏移

                # 生成解剖学分类 - 调整代数类型分布
                # 代数类型: 0=无气道, 1=气管, 2=主支气管, 3=叶支气管, 4=段支气管, 5=细支气管
                if branch_generation == 0:
                    generation_type = 1  # 气管
                elif branch_generation == 1:
                    generation_type = 2  # 主支气管
                elif branch_generation == 2 or branch_generation == 3:
                    generation_type = 3  # 叶支气管
                elif branch_generation == 4 or branch_generation == 5:
                    generation_type = 4  # 段支气管
                elif branch_generation == 6 or branch_generation == 7 or branch_generation == 8:
                    generation_type = 5  # 小支气管
                else:
                    generation_type = 6  # 细支气管

        # 判断是否包含分叉和结构类型 - 改进结构类型划分
        total_airway_volume = np.sum(parsing_patch > 0)
        branch_count = len(unique_branches)

        # 改进的结构复杂度指标计算
        if total_airway_volume > 0 and branch_count > 0:
            # 使用对数尺度处理体积，减少极值影响
            log_volume = np.log(total_airway_volume + 1) / np.log(10000)  # 归一化到合理范围

            # 考虑分支数的非线性增长关系
            normalized_branches = np.log(branch_count + 1) / np.log(10)

            # 复杂度现在结合了归一化的体积和分支数
            complexity = normalized_branches / max(0.1, log_volume)
        else:
            complexity = 0
        # # 结构复杂度指标：分支数量和体积的比例
        # complexity = branch_count / max(1, total_airway_volume / 1000)

        # 重新定义结构类型
        if branch_count == 0:
            structure_type = 0  # 无结构
        elif branch_count == 1:
            # 检查是否为细长管道
            # 计算主轴方向的长度与次轴方向长度的比值
            airway_mask = (parsing_patch > 0)
            if np.sum(airway_mask) > 0:
                # 计算气道区域的边界框
                z_indices, y_indices, x_indices = np.where(airway_mask)
                z_min, z_max = np.min(z_indices), np.max(z_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                x_min, x_max = np.min(x_indices), np.max(x_indices)

                # 计算三个维度的长度
                dimensions = [z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1]
                dimensions.sort()  # 排序，找出主轴和次轴

                # 如果最长轴与次长轴比例大于2，认为是细长管道
                if dimensions[2] / dimensions[1] > 2:
                    structure_type = 1  # 直管道
                else:
                    structure_type = 2  # 圆形/卵圆形管道
            else:
                structure_type = 1  # 默认为直管道
        elif complexity > 1.5:  # 复杂分叉区域
            structure_type = 6  # III级多分支复杂区域 1.5-
        elif complexity > 1:
            structure_type = 5  # II级多分支复杂区域 1-1.5
        elif complexity > 0.8:
            structure_type = 4  # I级多分支复杂区域 0.8-1
        else:  # 简单分叉
            structure_type = 3  # 分叉点

        # 空间位置分类 - 增加左右肺的区分
        z_center, y_center, x_center = [p + self.crop_size[i] // 2 for i, p in enumerate(position)]
        z_rel = z_center / img_shape[0]
        x_rel = x_center / img_shape[2]

        # 上/中/下肺分类
        if z_rel < 0.3:
            vertical_pos = 1  # 上肺
        elif z_rel < 0.6:
            vertical_pos = 2  # 中肺
        else:
            vertical_pos = 3  # 下肺

        # 左/右肺分类
        # 假设x轴从左到右，图像中点以右为右肺，以左为左肺
        if x_rel < 0.35:
            lateral_pos = 4  # 左肺
        elif x_rel > 0.65:
            lateral_pos = 5  # 右肺
        else:
            lateral_pos = 6  # 中央区域（通常包括气管）

        # 确定最终位置类型
        if generation_type == 1:  # 如果是气管
            location_type = 6  # 气管区域
        elif lateral_pos == 6:  # 中央区域非气管
            location_type = vertical_pos  # 使用垂直位置
        else:
            location_type = lateral_pos  # 使用左右位置

        airway_density = airway_pixels / (self.crop_size[0] * self.crop_size[1] * self.crop_size[2])

        # 计算纹理特征 - 可用于进一步区分气道类型
        if airway_pixels > 0:
            # 计算气道内CT值的标准差作为纹理复杂度指标
            airway_mask = (label_patch > 0)
            intensity_std = np.std(img_patch[airway_mask])

            # 计算气道壁厚度的估计值
            # 使用边界像素与内部像素的比值作为壁厚的粗略估计
            boundary_pixels = np.sum(parsing_patch == 1)  # 假设边界像素为1
            inner_pixels = airway_pixels - boundary_pixels
            wall_ratio = boundary_pixels / max(1, inner_pixels)
        else:
            intensity_std = 0
            wall_ratio = 0

        return {
            'airway_pixel_num': airway_pixels,
            'generation_type': generation_type,
            'structure_type': structure_type,
            'location_type': location_type,
            'contains_bifurcation': branch_count > 1,
            'airway_density': airway_density,
            'main_branch_id': main_branch,
            'branch_diversity': branch_count,
            'complexity': complexity,
            'intensity_std': intensity_std,
            'wall_ratio': wall_ratio
        }

    def process_case(self, case_id, mode="train"):
        """处理单个病例"""
        # 加载图像和标签
        img_path = os.path.join(self.raw_data_dir, mode, "image", f"{case_id}.nii.gz")
        label_path = os.path.join(self.raw_data_dir, mode, "label", f"{case_id}.nii.gz")

        try:
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            label_arr = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        except Exception as e:
            print(f"Error loading image/label for {case_id}: {e}")
            return

        # 加载解析数据
        parsing_arr, generation = self._load_parsing_data(case_id, mode)

        if parsing_arr is None:
            print(f"Skipping {case_id} due to missing parsing data")
            return

        # 计算裁剪位置
        positions = self._calculate_positions(img_arr.shape)

        for block_id, (z, y, x) in enumerate(positions):
            # 提取块数据
            img_patch = img_arr[z:z + self.crop_size[0], y:y + self.crop_size[1], x:x + self.crop_size[2]]
            label_patch = label_arr[z:z + self.crop_size[0], y:y + self.crop_size[1], x:x + self.crop_size[2]]
            parsing_patch = parsing_arr[z:z + self.crop_size[0], y:y + self.crop_size[1], x:x + self.crop_size[2]]

            # 计算解剖学特征 - 修正了参数列表
            anatomical_features = self._compute_anatomical_features(
                img_patch, label_patch, parsing_patch, generation, (z, y, x), img_arr.shape)

            # 构建元数据
            key = f"BAS{mode}_{case_id}_{block_id}"
            self.data_dict[key] = {
                'image_path': img_path,
                'label_path': label_path,
                'position': (z, y, x),
                'image_size': img_arr.shape,
                'crop_size': self.crop_size,
                'stride': self.stride,
                **anatomical_features
            }

    def generate_metadata(self, mode="train"):
        """生成所有病例的元数据"""
        case_ids = self._get_case_ids(mode)

        for case_id in tqdm(case_ids, desc=f"Processing {mode} cases"):
            self.process_case(case_id, mode)

        # 保存元数据
        with open(self.output_pkl_path, 'wb') as f:
            pickle.dump(self.data_dict, f)
        print(f"Metadata saved to {self.output_pkl_path}")

        # 打印分类分布统计
        self._print_statistics()

    # def _print_statistics(self):
    #     """打印数据集统计信息"""
    #     if not self.data_dict:
    #         print("No data to analyze")
    #         return
    #
    #     # 计数各类型数量
    #     gen_types = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    #     struct_types = {0: 0, 1: 0, 2: 0, 3: 0}
    #     loc_types = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    #
    #     for key, item in self.data_dict.items():
    #         gen_types[item.get('generation_type', 0)] += 1
    #         struct_types[item.get('structure_type', 0)] += 1
    #         loc_types[item.get('location_type', 0)] += 1
    #
    #     print("\n=== Dataset Statistics ===")
    #
    #     print("\nGeneration Type Distribution:")
    #     gen_names = {0: "无气道", 1: "气管", 2: "主支气管", 3: "叶支气管", 4: "段支气管", 5: "细支气管"}
    #     for gen_type, count in gen_types.items():
    #         print(f"  {gen_names[gen_type]}: {count} ({count / len(self.data_dict) * 100:.1f}%)")
    #
    #     print("\nStructure Type Distribution:")
    #     struct_names = {0: "无结构", 1: "直管道", 2: "分叉点", 3: "多分支区域"}
    #     for struct_type, count in struct_types.items():
    #         print(f"  {struct_names[struct_type]}: {count} ({count / len(self.data_dict) * 100:.1f}%)")
    #
    #     print("\nLocation Type Distribution:")
    #     loc_names = {0: "未定位", 1: "上肺", 2: "中肺", 3: "下肺", 4: "左肺", 5: "右肺", 6: "气管区域"}
    #     for loc_type, count in loc_types.items():
    #         print(f"  {loc_names[loc_type]}: {count} ({count / len(self.data_dict) * 100:.1f}%)")
    #
    #     print(f"\nTotal samples: {len(self.data_dict)}")

    def _print_statistics(self):
        """打印数据集统计信息"""
        if not self.data_dict:
            print("No data to analyze")
            return

        # 计数各类型数量
        gen_types = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,6:0}
        struct_types = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,6:0}
        loc_types = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        # 存储密度和复杂度的统计信息
        density_stats = []
        complexity_stats = []
        intensity_std_stats = []
        wall_ratio_stats = []

        for key, item in self.data_dict.items():
            gen_types[item.get('generation_type', 0)] += 1
            struct_types[item.get('structure_type', 0)] += 1
            loc_types[item.get('location_type', 0)] += 1

            # 收集连续特征的统计信息
            if item.get('airway_pixel_num', 0) > 0:  # 只统计有气道的样本
                density_stats.append(item.get('airway_density', 0))
                complexity_stats.append(item.get('complexity', 0))
                intensity_std_stats.append(item.get('intensity_std', 0))
                wall_ratio_stats.append(item.get('wall_ratio', 0))

        print("\n=== Dataset Statistics ===")

        print("\nGeneration Type Distribution:")
        gen_names = {0: "无气道", 1: "气管", 2: "主支气管", 3: "叶支气管", 4: "段支气管", 5: "小支气管", 6: "细支气管"}
        for gen_type, count in gen_types.items():
            print(f"  {gen_names[gen_type]}: {count} ({count / len(self.data_dict) * 100:.1f}%)")

        print("\nStructure Type Distribution:")
        struct_names = {0: "无结构", 1: "直管道", 2: "圆形管道", 3: "简单分叉",4:"I级多分支复杂区域",5:"II级多分支复杂区域",6:"III级多分支复杂区域"}  # 更新结构类型名称
        for struct_type, count in struct_types.items():
            print(f"  {struct_names[struct_type]}: {count} ({count / len(self.data_dict) * 100:.1f}%)")

        print("\nLocation Type Distribution:")
        loc_names = {0: "未定位", 1: "上肺", 2: "中肺", 3: "下肺", 4: "左肺", 5: "右肺", 6: "气管区域"}
        for loc_type, count in loc_types.items():
            print(f"  {loc_names[loc_type]}: {count} ({count / len(self.data_dict) * 100:.1f}%)")

        # 打印连续特征的统计信息
        if density_stats:
            print("\nContinuous Features Statistics (有气道样本):")
            print(
                f"  气道密度 - 平均值: {np.mean(density_stats):.4f}, 中位数: {np.median(density_stats):.4f}, 最大值: {np.max(density_stats):.4f}")
            print(
                f"  结构复杂度 - 平均值: {np.mean(complexity_stats):.4f}, 中位数: {np.median(complexity_stats):.4f}, 最大值: {np.max(complexity_stats):.4f}")
            print(
                f"  纹理标准差 - 平均值: {np.mean(intensity_std_stats):.4f}, 中位数: {np.median(intensity_std_stats):.4f}")
            print(f"  壁厚比例 - 平均值: {np.mean(wall_ratio_stats):.4f}, 中位数: {np.median(wall_ratio_stats):.4f}")

        # 打印有气道样本和无气道样本的比例
        airway_samples = len(self.data_dict) - gen_types[0]
        print(f"\n有气道样本: {airway_samples} ({airway_samples / len(self.data_dict) * 100:.1f}%)")
        print(f"无气道样本: {gen_types[0]} ({gen_types[0] / len(self.data_dict) * 100:.1f}%)")

        print(f"\nTotal samples: {len(self.data_dict)}")


# 执行入口
if __name__ == "__main__":
    generator = AirwayMetadataGenerator(
        raw_data_dir="/home/ubuntu/shared-data/BAS",
        parsing_dir="/home/ubuntu/shared-data/BAS/train/train_parsing_path",
        output_pkl_path="/home/ubuntu/dataset_info/airway_metadata_train_anatomical.pkl",
        crop_size=(128, 128, 128),
        stride=(64, 64, 64)
    )
    generator.generate_metadata(mode="train")
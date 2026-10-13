# BAS_precrop_getinfo.py
import os
import numpy as np
import pickle
import edt
import SimpleITK as sitk
from tqdm import tqdm

class BaseMetadataGenerator:
    def __init__(self,
                 raw_data_dir,
                 output_pkl_path,
                 crop_size=(128, 128, 128),
                 stride=(64, 64, 64)):
        self.raw_data_dir = raw_data_dir
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

    def _compute_metrics(self, img_patch, label_patch):
        """计算统计指标（包含三个关键指标）"""
        airway_pixels = np.sum(label_patch)
        edt_result = edt.edt(label_patch.astype(np.uint32), parallel=1)

        # 基础统计
        airway_region = (label_patch == 1)
        non_airway_region = (label_patch != 1)
        boundary_region = (edt_result == 1)
        inner_region = (edt_result > 1)

        airway_values = img_patch[airway_region]
        non_airway_values = img_patch[non_airway_region]
        boundary_values = img_patch[boundary_region]
        inner_values = img_patch[inner_region]

        # 计算三个关键指标
        boundary_inner_ratio = (
            len(boundary_values) / len(inner_values)
            if len(inner_values) > 0 else np.nan
        )
        boundary_inner_contrast = (
            np.mean(boundary_values) - np.mean(inner_values)
            if len(boundary_values) > 0 and len(inner_values) > 0 else np.nan
        )
        airway_non_airway_mean_deta = (
            np.mean(airway_values) - np.mean(non_airway_values)
            if len(airway_values) > 0 and len(non_airway_values) > 0 else np.nan
        )

        # 返回完整统计指标
        return {
            'airway_pixel_num': airway_pixels,
            'airway_pixel_num_boundary': len(boundary_values),
            'airway_pixel_num_inner': len(inner_values),
            'whole_strength_sum': np.sum(img_patch),
            'whole_strength_mean': np.mean(img_patch),
            'whole_strength_var': np.var(img_patch),
            'non_airway_strength_mean': np.mean(non_airway_values),
            'non_airway_strength_var': np.var(non_airway_values),
            'airway_strength_sum': np.sum(airway_values),
            'airway_strength_mean': np.mean(airway_values),
            'airway_strength_var': np.var(airway_values),
            'boundary_strength_mean': np.mean(boundary_values) if len(boundary_values) > 0 else np.nan,
            'inner_strength_mean': np.mean(inner_values) if len(inner_values) > 0 else np.nan,
            'boundary_inner_ratio': boundary_inner_ratio,
            'boundary_inner_contrast': boundary_inner_contrast,
            'airway_non_airway_mean_deta': airway_non_airway_mean_deta
        }


class BASGenerator(BaseMetadataGenerator):
    def __init__(self, mode="train", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode  # train/test

    def _get_case_ids(self):
        # 获取正确的图像目录
        image_dir = os.path.join(self.raw_data_dir, self.mode, "image")
        # 动态获取case_ids
        case_files = os.listdir(image_dir)
        case_ids = [f.split(".")[0] for f in case_files]
        return case_ids

    def _get_case_paths(self, case_id):
        """路径示例：
        /home/ubuntu/shared-data/BRONATLAS/train/image/CASE01-image.nii.gz
        /home/ubuntu/shared-data/BRONATLAS/train/label/CASE01-label.nii.gz
        """
        img_path = os.path.join(
            self.raw_data_dir, self.mode, self.mode+"_parsing_path", f"{case_id}_img.nii.gz"
        )
        label_path = os.path.join(
            self.raw_data_dir, self.mode, self.mode+"_parsing_path", f"{case_id}_label.nii.gz"
        )
        skel_path = ""  # 该数据集暂无skeleton
        return img_path, label_path, skel_path

    def _load_niigz_pair(self, case_id):
        """加载单个病例的niigz文件"""
        img_path = os.path.join(self.raw_data_dir, "image", f"{case_id}.nii.gz")
        label_path = os.path.join(self.raw_data_dir, "label", f"{case_id}.nii.gz")
        skel_path = os.path.join(self.raw_data_dir, self.mode+"_label_skeleton3", f"{case_id}.nii.gz")
        return (
            sitk.ReadImage(img_path),
            sitk.ReadImage(label_path),
            sitk.ReadImage(skel_path),
            img_path,  # 显式记录路径
            label_path,
            skel_path
        )

    def process_case(self, case_id):
        """处理单个病例"""
        img_path, label_path, skel_path = self._get_case_paths(case_id)
        # img_itk, label_itk, skel_itk,img_path, label_path, skel_path = self._load_niigz_pair(case_id)
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        label_arr = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        # skel_arr = sitk.GetArrayFromImage(skel_itk)

        positions = self._calculate_positions(img_arr.shape)

        for block_id, (z, y, x) in enumerate(positions):
            # 提取块数据
            img_patch = img_arr[z:z + self.crop_size[0], y:y + self.crop_size[1], x:x + self.crop_size[2]]
            label_patch = label_arr[z:z + self.crop_size[0], y:y + self.crop_size[1], x:x + self.crop_size[2]]
            # skel_patch = skel_arr[z:z + self.crop_size[0], y:y + self.crop_size[1], x:x + self.crop_size[2]]

            # 构建元数据
            key = "BAS"+self.mode+f"_{case_id}_{block_id}"
            self.data_dict[key] = {
                'image_path': img_path,  # 直接使用文件路径
                'label_path': label_path,
                'skeleton_path': skel_path,
                'position': (z, y, x),
                'image_size': img_arr.shape,
                'crop_size': self.crop_size,
                'stride': self.stride,
                **self._compute_metrics(img_patch, label_patch)
            }

    def generate_pkl(self):
        """主执行函数"""
        # case_ids = [f.split(".")[0] for f in os.listdir(os.path.join(self.raw_data_dir, "image"))]
        case_ids = self._get_case_ids()

        for case_id in tqdm(case_ids, desc="Processing cases"):
            self.process_case(case_id)

        with open(self.output_pkl_path, 'wb') as f:
            pickle.dump(self.data_dict, f)
        print(f"Metadata saved to {self.output_pkl_path}")


# ================== 执行入口 ==================
if __name__ == "__main__":
    generator = BASGenerator(
        mode='train',
        raw_data_dir="/home/ubuntu/shared-data/BAS",
        output_pkl_path="dataset_info/3dataset_info_BAStrain_merged.pkl",
        crop_size=(128, 128, 128),
        stride=(64, 64, 64)
    )
    generator.generate_pkl()

    generator = BASGenerator(
        mode='test',
        raw_data_dir="/home/ubuntu/shared-data/BAS",
        output_pkl_path="dataset_info/3dataset_info_BAStest_merged.pkl",
        crop_size=(128, 128, 128),
        stride=(64, 64, 64)
    )
    generator.generate_pkl()
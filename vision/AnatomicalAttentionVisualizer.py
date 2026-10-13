import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage


class AnatomicalAttentionVisualizer:
    """
    气道解剖学解耦注意力可视化工具
    """
    def __init__(self, output_dir="./anatomical_attention"):
        """
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 为不同任务定义不同的颜色映射
        self.colormaps = {
            'generation': plt.cm.viridis,  # 气道代数 - 绿色到紫色
            'structure': plt.cm.plasma,    # 结构类型 - 黄色到紫色
            'location': plt.cm.magma       # 空间位置 - 黑色到黄色
        }
        
        # 创建类别名称映射
        self.class_names = {
            'generation': {
                0: "无气道", 
                1: "气管", 
                2: "主支气管", 
                3: "叶支气管", 
                4: "段支气管", 
                5: "小支气管",
                6: "细支气管"
            },
            'structure': {
                0: "无结构", 
                1: "直管道", 
                2: "圆形管道", 
                3: "简单分叉",
                4: "一级复杂区域", 
                5: "二级复杂区域", 
                6: "三级复杂区域"
            },
            'location': {
                0: "未定位", 
                1: "上肺", 
                2: "中肺", 
                3: "下肺", 
                4: "左肺", 
                5: "右肺", 
                6: "气管区域"
            }
        }
    
    def visualize_mpr_slices(self, ct_volume, attention_map, case_id, task, 
                            target_class=None, slice_indices=None, alpha=0.7,
                            highlight_thres=0.6):
        """
        使用多平面重建(MPR)可视化注意力图
        
        Args:
            ct_volume: 输入CT体积 [B, C, D, H, W]
            attention_map: 注意力图 [B, 1, D, H, W]
            case_id: 样本ID
            task: 任务名称('generation', 'structure', 'location')
            target_class: 目标类别
            slice_indices: 指定切片索引，格式为{'axial': z, 'coronal': y, 'sagittal': x}
            alpha: 注意力图透明度
            highlight_thres: 高亮显示阈值
            
        Returns:
            str: 保存的图像路径
        """
        # 转换张量为NumPy数组
        if isinstance(ct_volume, torch.Tensor):
            ct_volume = ct_volume.detach().cpu().numpy()
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.detach().cpu().numpy()
        
        # 移除批次和通道维度
        ct_volume = ct_volume[0, 0]
        attention_map = attention_map[0, 0]
        
        # 如果未指定切片索引，找到最佳切片
        if slice_indices is None:
            slice_indices = self.find_optimal_slices(attention_map)
        
        # 获取任务对应的颜色映射
        cmap = self.colormaps.get(task, plt.cm.jet)
        
        # 设置三个平面的子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 获取窗宽窗位 - 假设CT范围为[-1000, 400] HU
        # 转换为显示范围[0, 1]
        vmin, vmax = -1000, 400  # HU值范围
        ct_volume_display = np.clip((ct_volume - vmin) / (vmax - vmin), 0, 1)
        
        # 创建二值化掩码用于高亮显示高注意力区域
        highlight_mask = attention_map > highlight_thres
        
        # 轴状面(z平面)
        z_slice = slice_indices['axial']
        axes[0].imshow(ct_volume_display[z_slice], cmap='gray')
        axes[0].imshow(attention_map[z_slice], cmap=cmap, alpha=alpha)
        
        # 添加高亮边缘
        if np.any(highlight_mask[z_slice]):
            # 找到高亮区域的边缘
            edges = ndimage.binary_dilation(highlight_mask[z_slice]) ^ highlight_mask[z_slice]
            # 在边缘上绘制红色轮廓
            axes[0].imshow(np.ma.masked_where(~edges, edges), cmap='autumn', alpha=1.0)
        
        axes[0].set_title(f'轴状面 (z={z_slice})')
        axes[0].axis('off')
        
        # 冠状面(y平面)
        y_slice = slice_indices['coronal']
        axes[1].imshow(ct_volume_display[:, y_slice, :], cmap='gray')
        axes[1].imshow(attention_map[:, y_slice, :], cmap=cmap, alpha=alpha)
        
        # 添加高亮边缘
        if np.any(highlight_mask[:, y_slice, :]):
            edges = ndimage.binary_dilation(highlight_mask[:, y_slice, :]) ^ highlight_mask[:, y_slice, :]
            axes[1].imshow(np.ma.masked_where(~edges, edges), cmap='autumn', alpha=1.0)
            
        axes[1].set_title(f'冠状面 (y={y_slice})')
        axes[1].axis('off')
        
        # 矢状面(x平面)
        x_slice = slice_indices['sagittal']
        axes[2].imshow(ct_volume_display[:, :, x_slice], cmap='gray')
        axes[2].imshow(attention_map[:, :, x_slice], cmap=cmap, alpha=alpha)
        
        # 添加高亮边缘
        if np.any(highlight_mask[:, :, x_slice]):
            edges = ndimage.binary_dilation(highlight_mask[:, :, x_slice]) ^ highlight_mask[:, :, x_slice]
            axes[2].imshow(np.ma.masked_where(~edges, edges), cmap='autumn', alpha=1.0)
            
        axes[2].set_title(f'矢状面 (x={x_slice})')
        axes[2].axis('off')
        
        # 添加整体标题
        class_name = self.class_names[task].get(target_class, f"类别{target_class}")
        fig.suptitle(f'{task}任务注意力图 - {class_name} - 样本ID: {case_id}')
        
        # 添加颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax)
        cbar.set_label('注意力强度')
        
        # 保存图像
        class_str = f"_class{target_class}" if target_class is not None else ""
        fig_path = os.path.join(self.output_dir, f'{case_id}_{task}{class_str}_mpr.png')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # 为颜色条留出空间
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig_path
    
    def find_optimal_slices(self, attention_map, label_map=None, percentile=90):
        """
        找到最佳可视化切片 - 注意力最高的位置
        
        Args:
            attention_map: 注意力图
            label_map: 可选的标签图来指导切片选择
            percentile: 百分位阈值，用于选择高注意力区域
            
        Returns:
            dict: 最佳切片索引
        """
        # 分别计算每个方向的最大投影
        z_projection = np.max(attention_map, axis=(1, 2))
        y_projection = np.max(attention_map, axis=(0, 2))
        x_projection = np.max(attention_map, axis=(0, 1))
        
        # 如果有标签图，与其组合来找到最佳切片
        if label_map is not None:
            if isinstance(label_map, torch.Tensor):
                label_map = label_map.detach().cpu().numpy()
            
            if len(label_map.shape) > 3:
                label_map = label_map[0, 0]  # 移除批次和通道维度
            
            # 使用标签作为掩码，只关注气道区域
            mask = label_map > 0
            
            # 仅在掩码内计算投影
            z_masked = np.zeros_like(z_projection)
            y_masked = np.zeros_like(y_projection)
            x_masked = np.zeros_like(x_projection)
            
            for z in range(attention_map.shape[0]):
                if np.any(mask[z]):
                    z_masked[z] = np.max(attention_map[z][mask[z]])
            
            for y in range(attention_map.shape[1]):
                if np.any(mask[:, y, :]):
                    y_masked[y] = np.max(attention_map[:, y, :][mask[:, y, :]])
            
            for x in range(attention_map.shape[2]):
                if np.any(mask[:, :, x]):
                    x_masked[x] = np.max(attention_map[:, :, x][mask[:, :, x]])
            
            # 只有在掩码区域有足够的非零值时才使用掩码投影
            if np.count_nonzero(z_masked) > attention_map.shape[0] * 0.1:
                z_projection = z_masked
            if np.count_nonzero(y_masked) > attention_map.shape[1] * 0.1:
                y_projection = y_masked
            if np.count_nonzero(x_masked) > attention_map.shape[2] * 0.1:
                x_projection = x_masked
        
        # 找到最高注意力的切片
        z_idx = np.argmax(z_projection)
        y_idx = np.argmax(y_projection)
        x_idx = np.argmax(x_projection)
        
        return {
            'axial': z_idx,
            'coronal': y_idx,
            'sagittal': x_idx
        }
    
    def create_3d_visualization(self, ct_volume, attention_map, case_id, task,
                               target_class=None, threshold=0.6,
                               volume_path=None, show_mip=True):
        """
        创建3D可视化 - 包括MIP和体渲染(如果可用)
        
        Args:
            ct_volume: 输入CT体积
            attention_map: 注意力图
            case_id: 样本ID
            task: 任务名称
            target_class: 目标类别
            threshold: 注意力阈值
            volume_path: 保存体积数据的路径
            show_mip: 是否显示最大强度投影
            
        Returns:
            tuple: (mip_path, volume_path)
        """
        # 转换张量为NumPy数组
        if isinstance(ct_volume, torch.Tensor):
            ct_volume = ct_volume.detach().cpu().numpy()
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.detach().cpu().numpy()
        
        # 移除批次和通道维度
        ct_volume = ct_volume[0, 0]
        attention_map = attention_map[0, 0]
        
        # 获取任务对应的颜色映射
        cmap = self.colormaps.get(task, plt.cm.jet)
        
        # 保存为NIFTI体积
        if volume_path is None:
            class_str = f"_class{target_class}" if target_class is not None else ""
            volume_path = os.path.join(self.output_dir, f'{case_id}_{task}{class_str}_attention.nii.gz')
        
        # 创建NIFTI图像
        nifti_img = nib.Nifti1Image(attention_map, np.eye(4))
        nib.save(nifti_img, volume_path)
        
        # 创建最大强度投影(MIP)
        if show_mip:
            mip_path = self._create_mip_visualization(attention_map, case_id, task, target_class, cmap)
            return mip_path, volume_path
        
        return None, volume_path
    
    def _create_mip_visualization(self, attention_map, case_id, task, target_class=None, cmap=None):
        """
        创建最大强度投影可视化
        
        Args:
            attention_map: 注意力图
            case_id: 样本ID
            task: 任务名称
            target_class: 目标类别
            cmap: 颜色映射
            
        Returns:
            str: 保存的MIP图像路径
        """
        if cmap is None:
            cmap = self.colormaps.get(task, plt.cm.jet)
        
        # 计算三个方向的MIP
        axial_mip = np.max(attention_map, axis=0)     # Z方向
        coronal_mip = np.max(attention_map, axis=1)   # Y方向
        sagittal_mip = np.max(attention_map, axis=2)  # X方向
        
        # 创建有三个子图的图像
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 绘制MIP
        axes[0].imshow(axial_mip, cmap=cmap)
        axes[0].set_title('轴状MIP (俯视图)')
        axes[0].axis('off')
        
        axes[1].imshow(coronal_mip, cmap=cmap)
        axes[1].set_title('冠状MIP (前视图)')
        axes[1].axis('off')
        
        axes[2].imshow(sagittal_mip, cmap=cmap)
        axes[2].set_title('矢状MIP (侧视图)')
        axes[2].axis('off')
        
        # 添加整体标题
        class_name = self.class_names[task].get(target_class, f"类别{target_class}")
        fig.suptitle(f'{task}任务MIP可视化 - {class_name} - 样本ID: {case_id}')
        
        # 添加颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax)
        cbar.set_label('注意力强度')
        
        # 保存图像
        class_str = f"_class{target_class}" if target_class is not None else ""
        mip_path = os.path.join(self.output_dir, f'{case_id}_{task}{class_str}_mip.png')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(mip_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return mip_path

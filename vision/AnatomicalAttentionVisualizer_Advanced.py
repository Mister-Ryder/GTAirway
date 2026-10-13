import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap


class AnatomicalAttentionAdvanced:
    """
    气道解剖学解耦注意力可视化的高级功能
    """
    def __init__(self, base_visualizer):
        """
        Args:
            base_visualizer: AnatomicalAttentionVisualizer的实例
        """
        self.base = base_visualizer
        self.output_dir = self.base.output_dir
    
    def overlay_on_original_slice(self, ct_volume, attention_map, label_map, case_id, task,
                                target_class=None, slice_index=None, axis=0, threshold=0.5):
        """
        在原始CT切片上叠加注意力图，并与分割标签对比
        
        Args:
            ct_volume: 输入CT体积
            attention_map: 注意力图
            label_map: 标签图
            case_id: 样本ID
            task: 任务名称
            target_class: 目标类别
            slice_index: 切片索引，如果为None则自动选择
            axis: 切片轴，0=轴状，1=冠状，2=矢状
            threshold: 注意力阈值
            
        Returns:
            str: 保存的图像路径
        """
        # 转换张量为NumPy数组
        if isinstance(ct_volume, torch.Tensor):
            ct_volume = ct_volume.detach().cpu().numpy()
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.detach().cpu().numpy()
        if isinstance(label_map, torch.Tensor):
            label_map = label_map.detach().cpu().numpy()
        
        # 移除批次和通道维度
        ct_volume = ct_volume[0, 0]
        attention_map = attention_map[0, 0]
        label_map = label_map[0, 0]
        
        # 如果未指定切片索引，找到最佳切片
        if slice_index is None:
            # 找到有气道的切片
            if axis == 0:  # 轴状
                airway_slices = np.where(np.sum(label_map > 0, axis=(1, 2)) > 0)[0]
                attention_sum = np.array([np.sum(attention_map[i]) for i in airway_slices])
                slice_index = airway_slices[np.argmax(attention_sum)] if len(airway_slices) > 0 else attention_map.shape[0] // 2
            elif axis == 1:  # 冠状
                airway_slices = np.where(np.sum(label_map > 0, axis=(0, 2)) > 0)[0]
                attention_sum = np.array([np.sum(attention_map[:, i, :]) for i in airway_slices])
                slice_index = airway_slices[np.argmax(attention_sum)] if len(airway_slices) > 0 else attention_map.shape[1] // 2
            else:  # 矢状
                airway_slices = np.where(np.sum(label_map > 0, axis=(0, 1)) > 0)[0]
                attention_sum = np.array([np.sum(attention_map[:, :, i]) for i in airway_slices])
                slice_index = airway_slices[np.argmax(attention_sum)] if len(airway_slices) > 0 else attention_map.shape[2] // 2
        
        # 创建两个子图的图像（原始+标签，原始+注意力）
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 选择对应轴的切片
        if axis == 0:  # 轴状
            ct_slice = ct_volume[slice_index]
            label_slice = label_map[slice_index]
            attention_slice = attention_map[slice_index]
            slice_name = f"轴状(z={slice_index})"
        elif axis == 1:  # 冠状
            ct_slice = ct_volume[:, slice_index, :]
            label_slice = label_map[:, slice_index, :]
            attention_slice = attention_map[:, slice_index, :]
            slice_name = f"冠状(y={slice_index})"
        else:  # 矢状
            ct_slice = ct_volume[:, :, slice_index]
            label_slice = label_map[:, :, slice_index]
            attention_slice = attention_map[:, :, slice_index]
            slice_name = f"矢状(x={slice_index})"
        
        # 设置窗宽窗位
        vmin, vmax = -1000, 400  # HU值范围
        ct_display = np.clip((ct_slice - vmin) / (vmax - vmin), 0, 1)
        
        # 左侧子图：CT+标签
        axes[0].imshow(ct_display, cmap='gray')
        # 使用红色显示气道标签
        label_mask = label_slice > 0
        overlay = np.zeros((*ct_slice.shape, 4))
        overlay[label_mask, 0] = 1.0  # 红色
        overlay[label_mask, 3] = 0.7  # 透明度
        axes[0].imshow(overlay)
        axes[0].set_title(f'CT与气道标签 - {slice_name}')
        axes[0].axis('off')
        
        # 右侧子图：CT+注意力
        axes[1].imshow(ct_display, cmap='gray')
        # 使用任务对应的颜色映射显示注意力
        cmap = self.base.colormaps.get(task, plt.cm.jet)
        attention_mask = attention_slice > threshold
        im = axes[1].imshow(attention_slice, cmap=cmap, alpha=0.7)
        axes[1].set_title(f'CT与{task}注意力 - {slice_name}')
        axes[1].axis('off')
        
        # 添加颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('注意力强度')
        
        # 添加整体标题
        class_name = self.base.class_names[task].get(target_class, f"类别{target_class}")
        fig.suptitle(f'{task}解剖学属性注意力 - {class_name} - 样本ID: {case_id}')
        
        # 保存图像
        class_str = f"_class{target_class}" if target_class is not None else ""
        axis_names = ["axial", "coronal", "sagittal"]
        overlay_path = os.path.join(self.output_dir, 
                                  f'{case_id}_{task}{class_str}_{axis_names[axis]}_overlay.png')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return overlay_path
    
    def compare_attention_maps(self, ct_volume, attention_maps, case_id,
                              slice_indices=None, alpha=0.7):
        """
        比较不同解剖属性的注意力图
        
        Args:
            ct_volume: 输入CT体积
            attention_maps: 字典，键为任务名称，值为注意力图
            case_id: 样本ID
            slice_indices: 指定切片索引
            alpha: 注意力图透明度
            
        Returns:
            str: 保存的比较图像路径
        """
        # 转换张量为NumPy数组
        if isinstance(ct_volume, torch.Tensor):
            ct_volume = ct_volume.detach().cpu().numpy()
        
        # 移除批次和通道维度
        ct_volume = ct_volume[0, 0]
        
        # 处理注意力图
        processed_maps = {}
        for task, attn_map in attention_maps.items():
            if isinstance(attn_map, torch.Tensor):
                processed_maps[task] = attn_map.detach().cpu().numpy()[0, 0]
            else:
                processed_maps[task] = attn_map
        
        # 如果未指定切片索引，找到最佳切片
        if slice_indices is None:
            # 合并所有注意力图来找到最佳切片
            combined_map = np.zeros_like(processed_maps[list(processed_maps.keys())[0]])
            for map_data in processed_maps.values():
                combined_map += map_data
            
            slice_indices = self.base.find_optimal_slices(combined_map)
        
        # 设置窗宽窗位
        vmin, vmax = -1000, 400  # HU值范围
        ct_display = np.clip((ct_volume - vmin) / (vmax - vmin), 0, 1)
        
        # 创建布局：每行表示一个平面，每列表示一种注意力类型
        tasks = list(processed_maps.keys())
        planes = ['axial', 'coronal', 'sagittal']
        
        fig, axes = plt.subplots(len(planes), len(tasks) + 1, figsize=(4 * (len(tasks) + 1), 12))
        
        # 绘制每个平面的CT图像（第一列）
        for i, plane in enumerate(planes):
            if plane == 'axial':
                slice_idx = slice_indices['axial']
                ct_slice = ct_display[slice_idx]
                axes[i, 0].imshow(ct_slice, cmap='gray')
                axes[i, 0].set_title(f'CT - 轴状面 (z={slice_idx})')
            elif plane == 'coronal':
                slice_idx = slice_indices['coronal']
                ct_slice = ct_display[:, slice_idx, :]
                axes[i, 0].imshow(ct_slice, cmap='gray')
                axes[i, 0].set_title(f'CT - 冠状面 (y={slice_idx})')
            elif plane == 'sagittal':
                slice_idx = slice_indices['sagittal']
                ct_slice = ct_display[:, :, slice_idx]
                axes[i, 0].imshow(ct_slice, cmap='gray')
                axes[i, 0].set_title(f'CT - 矢状面 (x={slice_idx})')
            
            axes[i, 0].axis('off')
        
        # 绘制每个任务的注意力图（后续列）
        for j, task in enumerate(tasks):
            cmap = self.base.colormaps.get(task, plt.cm.jet)
            
            for i, plane in enumerate(planes):
                if plane == 'axial':
                    slice_idx = slice_indices['axial']
                    ct_slice = ct_display[slice_idx]
                    attn_slice = processed_maps[task][slice_idx]
                    axes[i, j + 1].imshow(ct_slice, cmap='gray')
                    im = axes[i, j + 1].imshow(attn_slice, cmap=cmap, alpha=alpha)
                    axes[i, j + 1].set_title(f'{task} - 轴状面')
                elif plane == 'coronal':
                    slice_idx = slice_indices['coronal']
                    ct_slice = ct_display[:, slice_idx, :]
                    attn_slice = processed_maps[task][:, slice_idx, :]
                    axes[i, j + 1].imshow(ct_slice, cmap='gray')
                    im = axes[i, j + 1].imshow(attn_slice, cmap=cmap, alpha=alpha)
                    axes[i, j + 1].set_title(f'{task} - 冠状面')
                elif plane == 'sagittal':
                    slice_idx = slice_indices['sagittal']
                    ct_slice = ct_display[:, :, slice_idx]
                    attn_slice = processed_maps[task][:, :, slice_idx]
                    axes[i, j + 1].imshow(ct_slice, cmap='gray')
                    im = axes[i, j + 1].imshow(attn_slice, cmap=cmap, alpha=alpha)
                    axes[i, j + 1].set_title(f'{task} - 矢状面')
                
                axes[i, j + 1].axis('off')
                
                # 为每个子图添加颜色条
                divider = make_axes_locatable(axes[i, j + 1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
        
        # 添加整体标题
        fig.suptitle(f'解剖学属性注意力对比 - 样本ID: {case_id}', fontsize=16)
        
        # 保存图像
        comparison_path = os.path.join(self.output_dir, f'{case_id}_attention_comparison.png')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison_path
    
    def visualize_feature_maps(self, feature_maps, case_id, layer_name, 
                             max_features=16, slice_index=None):
        """
        可视化特定层的特征图
        
        Args:
            feature_maps: 特征图张量 [B, C, D, H, W]
            case_id: 样本ID
            layer_name: 层名称
            max_features: 最大可视化的特征数
            slice_index: 切片索引，如果为None则使用中间切片
            
        Returns:
            str: 保存的图像路径
        """
        # 转换张量为NumPy数组
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = feature_maps.detach().cpu().numpy()
        
        # 获取张量维度
        b, c, d, h, w = feature_maps.shape
        
        # 限制特征数量
        c = min(c, max_features)
        
        # 如果未指定切片索引，使用中间切片
        if slice_index is None:
            slice_index = d // 2
        
        # 创建网格布局
        grid_size = int(np.ceil(np.sqrt(c)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
        
        if grid_size == 1:
            axes = np.array([axes])
        
        # 展平轴数组以便索引
        axes = axes.flatten()
        
        # 绘制每个特征通道
        for i in range(c):
            # 提取特征图
            feature = feature_maps[0, i, slice_index]
            
            # 归一化特征图
            feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
            
            # 绘制特征图
            axes[i].imshow(feature, cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'特征 {i+1}')
        
        # 隐藏未使用的子图
        for i in range(c, len(axes)):
            axes[i].axis('off')
        
        # 添加整体标题
        fig.suptitle(f'层 {layer_name} 特征图 - 样本ID: {case_id} - 切片: {slice_index}', fontsize=16)
        
        # 保存图像
        feature_path = os.path.join(self.output_dir, f'{case_id}_{layer_name}_features_z{slice_index}.png')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(feature_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return feature_path
    
    def create_attention_difference_map(self, attention_map1, attention_map2, 
                                      case_id, task1, task2, 
                                      slice_indices=None):
        """
        创建两个注意力图之间的差异图
        
        Args:
            attention_map1: 第一个注意力图
            attention_map2: 第二个注意力图
            case_id: 样本ID
            task1: 第一个任务名称
            task2: 第二个任务名称
            slice_indices: 指定切片索引
            
        Returns:
            str: 保存的图像路径
        """
        # 转换张量为NumPy数组
        if isinstance(attention_map1, torch.Tensor):
            attention_map1 = attention_map1.detach().cpu().numpy()[0, 0]
        if isinstance(attention_map2, torch.Tensor):
            attention_map2 = attention_map2.detach().cpu().numpy()[0, 0]
        
        # 确保注意力图有相同的形状
        assert attention_map1.shape == attention_map2.shape, "注意力图形状不匹配"
        
        # 计算差异图 (Map1 - Map2)
        diff_map = attention_map1 - attention_map2
        
        # 如果未指定切片索引，找到最佳切片
        if slice_indices is None:
            # 使用差异图的绝对值来找到差异最明显的切片
            abs_diff = np.abs(diff_map)
            slice_indices = self.base.find_optimal_slices(abs_diff)
        
        # 设置三个平面的子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 创建红蓝颜色映射，中间为白色
        # 红色表示Map1更强，蓝色表示Map2更强
        colors = plt.cm.RdBu_r
        
        # 轴状面(z平面)
        z_slice = slice_indices['axial']
        im0 = axes[0].imshow(diff_map[z_slice], cmap=colors, vmin=-1, vmax=1)
        axes[0].set_title(f'轴状面差异 (z={z_slice})')
        axes[0].axis('off')
        
        # 冠状面(y平面)
        y_slice = slice_indices['coronal']
        im1 = axes[1].imshow(diff_map[:, y_slice, :], cmap=colors, vmin=-1, vmax=1)
        axes[1].set_title(f'冠状面差异 (y={y_slice})')
        axes[1].axis('off')
        
        # 矢状面(x平面)
        x_slice = slice_indices['sagittal']
        im2 = axes[2].imshow(diff_map[:, :, x_slice], cmap=colors, vmin=-1, vmax=1)
        axes[2].set_title(f'矢状面差异 (x={x_slice})')
        axes[2].axis('off')
        
        # 添加颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im0, cax=cbar_ax)
        cbar.set_label(f'{task1} - {task2} 差异')
        
        # 添加整体标题
        fig.suptitle(f'注意力差异图: {task1} vs {task2} - 样本ID: {case_id}')
        
        # 保存图像
        diff_path = os.path.join(self.output_dir, f'{case_id}_{task1}_vs_{task2}_difference.png')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # 为颜色条留出空间
        plt.savefig(diff_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return diff_path
    
    def visualize_class_activation_strength(self, probs_dict, case_id):
        """
        可视化不同类别的激活强度
        
        Args:
            probs_dict: 任务到概率张量的字典
            case_id: 样本ID
            
        Returns:
            str: 保存的图像路径
        """
        # 创建多个子图
        fig, axes = plt.subplots(len(probs_dict), 1, figsize=(12, 4 * len(probs_dict)))
        
        if len(probs_dict) == 1:
            axes = [axes]
        
        # 为每个任务绘制条形图
        for i, (task, probs) in enumerate(probs_dict.items()):
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            
            # 如果形状是[batch_size, num_classes]，取第一个样本
            if len(probs.shape) == 2:
                probs = probs[0]
            
            # 获取类别名称
            class_names = [self.base.class_names[task].get(j, f"类别{j}") for j in range(len(probs))]
            
            # 绘制条形图
            bar_colors = plt.cm.viridis(np.linspace(0, 0.8, len(probs)))
            bars = axes[i].barh(class_names, probs, color=bar_colors)
            
            # 添加数值标签
            for bar, prob in zip(bars, probs):
                axes[i].text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{prob:.3f}', va='center')
            
            # 设置标题和标签
            axes[i].set_title(f'{task} 类别概率')
            axes[i].set_xlabel('概率')
            axes[i].set_xlim(0, 1.1)
            
            # 添加网格线
            axes[i].grid(axis='x', linestyle='--', alpha=0.7)
        
        # 添加整体标题
        fig.suptitle(f'解剖学属性分类概率 - 样本ID: {case_id}', fontsize=16)
        
        # 保存图像
        probs_path = os.path.join(self.output_dir, f'{case_id}_class_probabilities.png')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(probs_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return probs_path
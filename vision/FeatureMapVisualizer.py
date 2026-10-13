import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class AirwayFeatureMapVisualizer:
    """
    用于可视化AirwayAnatomicalClassifier中的特征图
    """
    def __init__(self, model, output_dir="./feature_maps"):
        """
        Args:
            model: AirwayAnatomicalClassifier模型
            output_dir: 输出目录
        """
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 储存钩子函数和特征
        self.hooks = []
        self.features = {}
    
    def _get_layer_by_name(self, name):
        """
        通过名称获取模型中的层
        
        Args:
            name: 层名称
            
        Returns:
            nn.Module: 找到的层
        """
        if name.startswith('encoder.'):
            idx = int(name.split('.')[1])
            return self.model.encoder[idx]
        elif name.startswith('attention_blocks.'):
            idx = int(name.split('.')[1])
            return self.model.attention_blocks[idx]
        elif name == 'guidance_head':
            return self.model.guidance_head
        else:
            parts = name.split('.')
            curr_module = self.model
            for part in parts:
                if part.isdigit():
                    curr_module = curr_module[int(part)]
                else:
                    curr_module = getattr(curr_module, part)
            return curr_module
    
    def register_hooks(self, layer_names):
        """
        为指定层注册钩子
        
        Args:
            layer_names: 层名称列表
            
        Returns:
            list: 钩子句柄列表
        """
        def hook_fn(name):
            def fn(module, input, output):
                self.features[name] = output.detach()
            return fn
        
        for name in layer_names:
            layer = self._get_layer_by_name(name)
            hook = layer.register_forward_hook(hook_fn(name))
            self.hooks.append(hook)
        
        return self.hooks
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_features(self, input_tensor):
        """
        提取特征图
        
        Args:
            input_tensor: 输入张量
            
        Returns:
            dict: 层名称到特征图的映射
        """
        # 确保模型处于评估模式
        self.model.eval()
        
        # 清空特征字典
        self.features = {}
        
        # 前向传播
        with torch.no_grad():
            self.model(input_tensor)
        
        return self.features
    
    def visualize_feature_map(self, feature_map, layer_name, case_id, 
                            slice_idx=None, max_features=16):
        """
        可视化特征图
        
        Args:
            feature_map: 特征图张量 [B, C, D, H, W]
            layer_name: 层名称
            case_id: 样本ID
            slice_idx: 切片索引，如果为None则使用中间切片
            max_features: 最大可视化的特征数
            
        Returns:
            str: 保存的图像路径
        """
        # 转换张量为NumPy数组
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.detach().cpu().numpy()
        
        # 获取张量维度
        b, c, d, h, w = feature_map.shape
        
        # 如果未指定切片索引，使用中间切片
        if slice_idx is None:
            slice_idx = d // 2
        
        # 限制特征通道数量
        c = min(c, max_features)
        
        # 计算网格大小
        grid_size = int(np.ceil(np.sqrt(c)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
        
        # 确保axes是数组
        if grid_size == 1:
            axes = np.array([axes])
        
        # 展平轴数组以便索引
        axes = axes.flatten()
        
        # 绘制每个特征通道
        for i in range(c):
            # 提取特征图
            feat = feature_map[0, i, slice_idx]
            
            # 归一化特征图以便可视化
            feat_min, feat_max = feat.min(), feat.max()
            if feat_max > feat_min:
                feat = (feat - feat_min) / (feat_max - feat_min)
            
            # 绘制特征图
            im = axes[i].imshow(feat, cmap='viridis')
            axes[i].set_title(f'通道 {i}')
            axes[i].axis('off')
            
            # 添加颜色条
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        
        # 隐藏未使用的子图
        for i in range(c, len(axes)):
            axes[i].axis('off')
        
        # 添加整体标题
        fig.suptitle(f'层 {layer_name} 特征图 (切片 z={slice_idx}) - 样本ID: {case_id}')
        
        # 保存图像
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f'{case_id}_{layer_name}_z{slice_idx}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_multiple_slices(self, feature_map, layer_name, case_id, 
                                slice_indices=None, channel_idx=0):
        """
        可视化单个通道的多个切片
        
        Args:
            feature_map: 特征图张量 [B, C, D, H, W]
            layer_name: 层名称
            case_id: 样本ID
            slice_indices: 切片索引列表，如果为None则均匀选择切片
            channel_idx: 要可视化的通道索引
            
        Returns:
            str: 保存的图像路径
        """
        # 转换张量为NumPy数组
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.detach().cpu().numpy()
        
        # 获取张量维度
        b, c, d, h, w = feature_map.shape
        
        # 如果未指定切片索引，均匀选择9个切片
        if slice_indices is None:
            num_slices = min(9, d)
            slice_indices = np.linspace(0, d-1, num_slices, dtype=int)
        
        # 计算网格大小
        grid_size = int(np.ceil(np.sqrt(len(slice_indices))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
        
        # 确保axes是数组
        if grid_size == 1:
            axes = np.array([axes])
        
        # 展平轴数组以便索引
        axes = axes.flatten()
        
        # 提取指定通道的特征图
        feat_channel = feature_map[0, channel_idx]
        
        # 归一化整个通道以便保持一致的色彩映射
        feat_min, feat_max = feat_channel.min(), feat_channel.max()
        if feat_max > feat_min:
            feat_channel = (feat_channel - feat_min) / (feat_max - feat_min)
        
        # 绘制每个切片
        for i, slice_idx in enumerate(slice_indices):
            if i < len(axes):
                im = axes[i].imshow(feat_channel[slice_idx], cmap='viridis')
                axes[i].set_title(f'切片 z={slice_idx}')
                axes[i].axis('off')
        
        # 隐藏未使用的子图
        for i in range(len(slice_indices), len(axes)):
            axes[i].axis('off')
        
        # 添加整体标题
        fig.suptitle(f'层 {layer_name} 通道 {channel_idx} 多切片视图 - 样本ID: {case_id}')
        
        # 添加颜色条
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('特征强度')
        
        # 保存图像
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, 
                                f'{case_id}_{layer_name}_channel{channel_idx}_multi_slice.png')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_guidance_map(self, guidance_map, case_id, slice_idx=None):
        """
        可视化解剖引导图
        
        Args:
            guidance_map: 引导图张量 [B, C, D, H, W]
            case_id: 样本ID
            slice_idx: 切片索引，如果为None则使用中间切片
            
        Returns:
            str: 保存的图像路径
        """
        # 转换张量为NumPy数组
        if isinstance(guidance_map, torch.Tensor):
            guidance_map = guidance_map.detach().cpu().numpy()
        
        # 获取张量维度
        b, c, d, h, w = guidance_map.shape
        
        # 如果未指定切片索引，使用中间切片
        if slice_idx is None:
            slice_idx = d // 2
        
        # 创建包含所有通道的可视化
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        
        # 绘制每个引导通道
        for i in range(min(c, 16)):
            # 提取引导图
            guide = guidance_map[0, i, slice_idx]
            
            # 归一化
            guide_min, guide_max = guide.min(), guide.max()
            if guide_max > guide_min:
                guide = (guide - guide_min) / (guide_max - guide_min)
            
            # 绘制引导图
            im = axes[i].imshow(guide, cmap='viridis')
            axes[i].set_title(f'引导通道 {i}')
            axes[i].axis('off')
            
            # 添加颜色条
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        
        # 隐藏未使用的子图
        for i in range(c, len(axes)):
            axes[i].axis('off')
        
        # 添加整体标题
        fig.suptitle(f'解剖引导图 (切片 z={slice_idx}) - 样本ID: {case_id}')
        
        # 保存图像
        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f'{case_id}_guidance_map_z{slice_idx}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def run_complete_visualization(self, input_tensor, case_id, 
                                  layers=None, slice_idx=None):
        """
        运行完整的特征可视化流程
        
        Args:
            input_tensor: 输入张量 [B, C, D, H, W]
            case_id: 样本ID
            layers: 要可视化的层列表，如果为None则使用默认层
            slice_idx: 切片索引，如果为None则使用中间切片
            
        Returns:
            dict: 包含各层可视化结果的字典
        """
        # 默认层列表
        if layers is None:
            layers = [
                'encoder.1',
                'encoder.3',
                'encoder.5',
                'encoder.7',
                'attention_blocks.0',
                'attention_blocks.1',
                'attention_blocks.2',
                'guidance_head'
            ]
        
        # 注册钩子
        self.register_hooks(layers)
        
        # 提取特征
        features = self.extract_features(input_tensor)
        
        # 获取中间切片索引（如果未指定）
        if slice_idx is None and len(input_tensor.shape) == 5:
            slice_idx = input_tensor.shape[2] // 2
        
        # 可视化结果
        results = {}
        
        # 可视化每一层的特征图
        for layer_name, feature_map in features.items():
            # 标准特征图可视化
            viz_path = self.visualize_feature_map(
                feature_map, layer_name, case_id, slice_idx=slice_idx
            )
            results[f'{layer_name}_features'] = viz_path
            
            # 如果是guidance_head，还需要可视化引导图
            if layer_name == 'guidance_head':
                guidance_path = self.visualize_guidance_map(
                    feature_map, case_id, slice_idx=slice_idx
                )
                results['guidance_map'] = guidance_path
            
            # 对encoder.5和attention_blocks.1层，可视化多切片视图
            if layer_name in ['encoder.5', 'attention_blocks.1']:
                # 为前两个通道创建多切片视图
                for channel_idx in range(min(2, feature_map.shape[1])):
                    multi_slice_path = self.visualize_multiple_slices(
                        feature_map, layer_name, case_id, channel_idx=channel_idx
                    )
                    results[f'{layer_name}_channel{channel_idx}_multi_slice'] = multi_slice_path
        
        # 移除钩子
        self.remove_hooks()
        
        return results
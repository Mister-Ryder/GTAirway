#!/usr/bin/env python3
# example_usage.py
"""
气道解剖学解耦注意力可视化工具使用示例
"""

import os
import torch
import matplotlib.pyplot as plt
import argparse

# 导入自定义模块
from AirwayAnatomicalGradCAM import AirwayAnatomicalGradCAM
from AnatomicalAttentionVisualizer import AnatomicalAttentionVisualizer
from AnatomicalAttentionVisualizer_Advanced import AnatomicalAttentionAdvanced
from FeatureMapVisualizer import AirwayFeatureMapVisualizer

# 导入模型和数据集
from func.AnatomicalClassifier import AirwayAnatomicalClassifier
from func.load_dataset import airway_dataset, PKLtoNPYAdapter


def parse_args():
    parser = argparse.ArgumentParser(description='气道解剖学注意力可视化示例')
    
    parser.add_argument('--pkl_path', type=str, required=True, help='PKL元数据文件路径')
    parser.add_argument('--npy_base_dir', type=str, required=True, help='NPY文件的基础目录')
    parser.add_argument('--model_path', type=str, required=True, help='预训练的分类器权重路径') 
    parser.add_argument('--output_dir', type=str, default='./example_output', help='输出目录')
    parser.add_argument('--case_id', type=str, default=None, help='要可视化的样本ID')
    
    args = parser.parse_args()
    return args


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据集
    print("Loading dataset...")
    adapter = PKLtoNPYAdapter(args.pkl_path, args.npy_base_dir)
    data_dict = adapter.get_adapted_dict()
    
    # 创建数据集
    dataset = airway_dataset(data_dict, add_anatomical_features=True, model_type="classifier")
    dataset.set_para(
        file_format='.npy',
        crop_size=(128, 128, 128),
        windowMin=-1000,
        windowMax=600,
        need_tensor_output=True,
        need_transform=False  # 不需要数据增强
    )
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # 加载模型
    print(f"Loading model from {args.model_path}")
    model = AirwayAnatomicalClassifier(
        in_channels=1,
        generation_classes=7,
        structure_classes=7,
        location_classes=7
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 处理DataParallel前缀
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 选择样本
    if args.case_id and args.case_id in dataset.name_list:
        case_idx = dataset.name_list.index(args.case_id)
    else:
        # 随机选择一个样本
        import numpy as np
        case_idx = np.random.randint(len(dataset))
        args.case_id = dataset.name_list[case_idx]
    
    print(f"Selected case: {args.case_id} (index {case_idx})")
    
    # 加载样本数据
    sample = dataset[case_idx]
    input_tensor = sample['image'].unsqueeze(0).to(device)  # 添加批次维度
    
    # 创建可视化输出目录
    case_output_dir = os.path.join(args.output_dir, args.case_id)
    os.makedirs(case_output_dir, exist_ok=True)
    
    # 1. 基本GradCAM可视化
    print("\n1. 基本GradCAM可视化")
    visualizer = AnatomicalAttentionVisualizer(output_dir=case_output_dir)
    
    # 获取目标层
    target_layer = model.encoder[5]
    
    # 为每个任务生成注意力图
    tasks = ['generation', 'structure', 'location']
    attention_maps = {}
    
    for task in tasks:
        # 创建GradCAM实例
        gradcam = AirwayAnatomicalGradCAM(model, target_layer, task=task)
        
        # 获取真实类别
        target_class = sample[f'{task}_type'].item()
        
        # 获取预测类别和概率
        pred_class, probabilities = gradcam.get_task_prediction(input_tensor)
        
        print(f"  {task.capitalize()} - 真实类别: {target_class}, 预测类别: {pred_class}")
        
        # 生成注意力图
        attention_map = gradcam.generate_cam(input_tensor, target_class=target_class)
        attention_maps[task] = attention_map
        
        # 可视化多平面切片
        mpr_path = visualizer.visualize_mpr_slices(
            input_tensor, attention_map, args.case_id, task, 
            target_class=target_class
        )
        print(f"  MPR visualization saved to: {mpr_path}")
        
        # 创建3D可视化
        mip_path, volume_path = visualizer.create_3d_visualization(
            input_tensor, attention_map, args.case_id, task, 
            target_class=target_class
        )
        print(f"  MIP visualization saved to: {mip_path}")
        print(f"  Volume saved to: {volume_path}")
        
        # 移除钩子
        gradcam.remove_hooks()
    
    # 2. 高级可视化
    print("\n2. 高级可视化")
    adv_vis = AnatomicalAttentionAdvanced(visualizer)
    
    # 比较不同任务的注意力图
    comparison_path = adv_vis.compare_attention_maps(
        input_tensor, attention_maps, args.case_id
    )
    print(f"  Attention comparison saved to: {comparison_path}")
    
    # 计算不同任务之间的差异图
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            task1, task2 = tasks[i], tasks[j]
            diff_path = adv_vis.create_attention_difference_map(
                attention_maps[task1],
                attention_maps[task2],
                args.case_id, task1, task2
            )
            print(f"  Difference map ({task1} vs {task2}) saved to: {diff_path}")
    
    # 3. 特征图可视化
    print("\n3. 特征图可视化")
    feature_viz = AirwayFeatureMapVisualizer(model, output_dir=os.path.join(case_output_dir, 'feature_maps'))
    
    # 定义要可视化的层
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
    
    # 运行完整的特征可视化
    feature_results = feature_viz.run_complete_visualization(input_tensor, args.case_id, layers=layers)
    
    print(f"  Feature map visualizations saved to: {os.path.join(case_output_dir, 'feature_maps')}")
    for layer, path in feature_results.items():
        print(f"    - {layer}: {os.path.basename(path)}")
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

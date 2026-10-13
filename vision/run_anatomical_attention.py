#!/usr/bin/env python3
# run_anatomical_attention.py
"""
运行气道解剖学注意力可视化
此脚本用于载入预训练的气道解剖学分类器，并生成各解剖属性(generation/structure/location)的注意力可视化
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
import pickle

# 导入自定义模块
from AirwayAnatomicalGradCAM import AirwayAnatomicalGradCAM
from AnatomicalAttentionVisualizer import AnatomicalAttentionVisualizer
from AnatomicalAttentionVisualizer_Advanced import AnatomicalAttentionAdvanced

# 导入模型和数据集
from func.AnatomicalClassifier import AirwayAnatomicalClassifier
from func.load_dataset import airway_dataset, PKLtoNPYAdapter


def parse_args():
    parser = argparse.ArgumentParser(description='气道解剖学解耦注意力可视化')
    
    # 数据路径
    parser.add_argument('--pkl_path', type=str, 
                       default='./dataset_info/airway_metadata_train_anatomical.pkl',
                       help='PKL元数据文件路径')
    parser.add_argument('--npy_base_dir', type=str,
                       default='/path/to/npy/files',
                       help='NPY文件的基础目录')
    
    # 模型路径
    parser.add_argument('--model_path', type=str, 
                       default='./runs_anatomical_classifier/airway_anatomical_classifier_best.pkl',
                       help='预训练的分类器权重路径')
    
    # 可视化设置
    parser.add_argument('--output_dir', type=str, 
                        default='./airway_attention_results',
                        help='输出目录')
    parser.add_argument('--mode', type=str, 
                        default='single',
                        choices=['single', 'batch', 'compare'],
                        help='可视化模式: single=单个样本,batch=批量样本,compare=比较模式')
    parser.add_argument('--case_ids', type=str, nargs='+',
                        default=None,
                        help='要可视化的样本ID列表')
    parser.add_argument('--num_samples', type=int, 
                        default=5,
                        help='批量模式下的样本数量')
                        
    # 注意力层选择
    parser.add_argument('--target_layer', type=str, 
                        default='encoder.5',
                        choices=['encoder.5', 'encoder.7', 'attention_blocks.1'],
                        help='目标注意力层')
    
    # 其他设置
    parser.add_argument('--gpu_id', type=int, 
                        default=0,
                        help='使用的GPU设备ID')
    parser.add_argument('--save_nifti', action='store_true',
                        help='是否保存NIFTI格式的注意力图')
    
    args = parser.parse_args()
    return args


def load_model(model_path, device):
    """
    加载预训练的气道解剖学分类器
    
    Args:
        model_path: 模型权重路径
        device: 使用的设备
        
    Returns:
        model: 加载好的模型
    """
    # 初始化模型
    model = AirwayAnatomicalClassifier(
        in_channels=1,
        generation_classes=7,
        structure_classes=7,
        location_classes=7
    ).to(device)
    
    # 加载权重
    print(f"Loading model weights from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理不同保存格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 处理DataParallel前缀
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def get_target_layer(model, layer_name):
    """
    获取目标注意力层
    
    Args:
        model: 分类器模型
        layer_name: 层名称
        
    Returns:
        target_layer: 目标层
    """
    if layer_name == 'encoder.5':
        return model.encoder[5]
    elif layer_name == 'encoder.7':
        return model.encoder[7]
    elif layer_name == 'attention_blocks.1':
        return model.attention_blocks[1]
    else:
        raise ValueError(f"Unknown layer: {layer_name}")


def run_single_case_visualization(model, dataset, case_idx, visualizer, adv_vis, device, args):
    """
    对单个样本进行注意力可视化
    
    Args:
        model: 分类器模型
        dataset: 数据集
        case_idx: 样本索引
        visualizer: 基础可视化器
        adv_vis: 高级可视化器
        device: 使用的设备
        args: 命令行参数
        
    Returns:
        dict: 可视化结果
    """
    # 获取样本数据
    sample = dataset[case_idx]
    case_id = dataset.name_list[case_idx]
    print(f"Processing case: {case_id}")
    
    # 准备输入数据
    input_tensor = sample['image'].unsqueeze(0).to(device)  # 添加批次维度
    
    # 获取目标层
    target_layer = get_target_layer(model, args.target_layer)
    
    # 对每个任务运行GradCAM
    tasks = ['generation', 'structure', 'location']
    results = {'case_id': case_id, 'inputs': input_tensor.cpu()}
    
    all_attention_maps = {}
    all_probs = {}
    
    for task in tasks:
        print(f"  Visualizing {task} attention")
        
        # 创建GradCAM实例
        gradcam = AirwayAnatomicalGradCAM(model, target_layer, task=task)
        
        # 获取真实类别
        target_class = sample[f'{task}_type'].item()
        results[f'{task}_class'] = target_class
        
        # 获取预测类别和概率
        pred_class, probabilities = gradcam.get_task_prediction(input_tensor)
        results[f'{task}_pred'] = pred_class
        results[f'{task}_probs'] = probabilities.cpu()
        all_probs[task] = probabilities.cpu()[0].numpy()
        
        # 生成注意力图
        attention_map = gradcam.generate_cam(input_tensor, target_class=target_class)
        results[f'{task}_attention'] = attention_map.cpu()
        all_attention_maps[task] = attention_map
        
        # 可视化多平面切片
        mpr_path = visualizer.visualize_mpr_slices(
            input_tensor, attention_map, case_id, task, 
            target_class=target_class
        )
        results[f'{task}_mpr_path'] = mpr_path
        print(f"    MPR visualization saved to: {mpr_path}")
        
        # 创建3D可视化
        if args.save_nifti:
            mip_path, volume_path = visualizer.create_3d_visualization(
                input_tensor, attention_map, case_id, task, 
                target_class=target_class
            )
            results[f'{task}_mip_path'] = mip_path
            results[f'{task}_volume_path'] = volume_path
            print(f"    MIP visualization saved to: {mip_path}")
            print(f"    Volume saved to: {volume_path}")
        
        # 移除钩子
        gradcam.remove_hooks()
    
    # 使用高级可视化器进行比较
    comparison_path = adv_vis.compare_attention_maps(
        input_tensor, 
        {task: results[f'{task}_attention'] for task in tasks},
        case_id
    )
    results['comparison_path'] = comparison_path
    print(f"  Attention comparison saved to: {comparison_path}")
    
    # 可视化类别激活强度
    probs_path = adv_vis.visualize_class_activation_strength(
        {task: results[f'{task}_probs'] for task in tasks},
        case_id
    )
    results['probs_path'] = probs_path
    print(f"  Class probabilities visualization saved to: {probs_path}")
    
    # 计算不同任务之间的差异图
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            task1, task2 = tasks[i], tasks[j]
            diff_path = adv_vis.create_attention_difference_map(
                results[f'{task1}_attention'],
                results[f'{task2}_attention'],
                case_id, task1, task2
            )
            results[f'{task1}_vs_{task2}_diff_path'] = diff_path
            print(f"  Difference map ({task1} vs {task2}) saved to: {diff_path}")
    
    return results


def run_batch_visualization(model, dataset, num_samples, visualizer, adv_vis, device, args):
    """
    对多个样本进行批量注意力可视化
    
    Args:
        model: 分类器模型
        dataset: 数据集
        num_samples: 样本数量
        visualizer: 基础可视化器
        adv_vis: 高级可视化器
        device: 使用的设备
        args: 命令行参数
        
    Returns:
        list: 可视化结果列表
    """
    # 如果提供了案例ID列表，优先使用
    if args.case_ids:
        case_indices = [dataset.name_list.index(case_id) for case_id in args.case_ids if case_id in dataset.name_list]
        print(f"Found {len(case_indices)} cases from the provided list")
    else:
        # 否则随机选择指定数量的样本
        total_samples = len(dataset)
        num_samples = min(num_samples, total_samples)
        case_indices = np.random.choice(total_samples, num_samples, replace=False)
        print(f"Randomly selected {len(case_indices)} cases")
    
    # 对每个样本进行可视化
    results = []
    for i, case_idx in enumerate(tqdm(case_indices, desc="Processing cases")):
        try:
            case_result = run_single_case_visualization(
                model, dataset, case_idx, visualizer, adv_vis, device, args
            )
            results.append(case_result)
        except Exception as e:
            print(f"Error processing case {dataset.name_list[case_idx]}: {e}")
            continue
            
    return results


def create_summary_report(results, output_dir):
    """
    创建汇总报告
    
    Args:
        results: 可视化结果列表
        output_dir: 输出目录
        
    Returns:
        str: 报告路径
    """
    # 创建Markdown报告
    report_path = os.path.join(output_dir, "attention_analysis_report.md")
    
    with open(report_path, "w") as f:
        f.write("# 气道解剖学解耦注意力分析报告\n\n")
        
        # 总结分析的案例数
        f.write(f"## 分析概况\n\n")
        f.write(f"- 分析案例总数: {len(results)}\n")
        
        # 统计各任务的分类准确率
        tasks = ['generation', 'structure', 'location']
        f.write("\n## 分类性能\n\n")
        
        for task in tasks:
            correct = sum(1 for r in results if r[f'{task}_class'] == r[f'{task}_pred'])
            accuracy = correct / len(results) if len(results) > 0 else 0
            f.write(f"- {task} 任务准确率: {accuracy:.2%} ({correct}/{len(results)})\n")
        
        # 各任务的类别分布
        f.write("\n## 类别分布\n\n")
        
        for task in tasks:
            f.write(f"### {task} 任务\n\n")
            
            # 统计真实类别分布
            class_counts = {}
            for r in results:
                cls = r[f'{task}_class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            f.write("| 类别 | 数量 | 百分比 |\n")
            f.write("|------|------|--------|\n")
            
            for cls in sorted(class_counts.keys()):
                count = class_counts[cls]
                percentage = count / len(results) if len(results) > 0 else 0
                f.write(f"| {cls} | {count} | {percentage:.2%} |\n")
            
            f.write("\n")
        
        # 注意力可视化示例
        f.write("\n## 注意力可视化示例\n\n")
        
        # 选择前3个案例作为示例
        for i, result in enumerate(results[:3]):
            case_id = result['case_id']
            f.write(f"### 案例 {i+1}: {case_id}\n\n")
            
            # 添加多平面可视化图像链接
            for task in tasks:
                mpr_path = result[f'{task}_mpr_path']
                rel_path = os.path.relpath(mpr_path, output_dir)
                f.write(f"#### {task} 任务注意力图\n\n")
                f.write(f"![{task} Attention]({rel_path})\n\n")
                f.write(f"- 真实类别: {result[f'{task}_class']}\n")
                f.write(f"- 预测类别: {result[f'{task}_pred']}\n\n")
            
            # 添加比较图链接
            comparison_path = result['comparison_path']
            rel_comparison_path = os.path.relpath(comparison_path, output_dir)
            f.write(f"#### 注意力对比\n\n")
            f.write(f"![Attention Comparison]({rel_comparison_path})\n\n")
            
            # 添加分类概率图链接
            probs_path = result['probs_path']
            rel_probs_path = os.path.relpath(probs_path, output_dir)
            f.write(f"#### 分类概率\n\n")
            f.write(f"![Classification Probabilities]({rel_probs_path})\n\n")
        
        # 包含所有分析案例的表格
        f.write("\n## 所有分析案例\n\n")
        f.write("| 案例ID | Generation真实/预测 | Structure真实/预测 | Location真实/预测 |\n")
        f.write("|--------|-------------------|-------------------|------------------|\n")
        
        for result in results:
            case_id = result['case_id']
            gen_real = result['generation_class']
            gen_pred = result['generation_pred']
            struct_real = result['structure_class']
            struct_pred = result['structure_pred']
            loc_real = result['location_class']
            loc_pred = result['location_pred']
            
            f.write(f"| {case_id} | {gen_real}/{gen_pred} | {struct_real}/{struct_pred} | {loc_real}/{loc_pred} |\n")
    
    print(f"Summary report saved to: {report_path}")
    return report_path


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    model = load_model(args.model_path, device)
    
    # 创建可视化器
    visualizer = AnatomicalAttentionVisualizer(output_dir=args.output_dir)
    adv_vis = AnatomicalAttentionAdvanced(visualizer)
    
    # 根据模式运行可视化
    if args.mode == 'single':
        # 单样本模式 - 使用第一个提供的案例ID或第一个样本
        if args.case_ids and args.case_ids[0] in dataset.name_list:
            case_idx = dataset.name_list.index(args.case_ids[0])
        else:
            case_idx = 0
        
        # 运行单样本可视化
        result = run_single_case_visualization(
            model, dataset, case_idx, visualizer, adv_vis, device, args
        )
        results = [result]
    
    elif args.mode == 'batch':
        # 批量模式
        results = run_batch_visualization(
            model, dataset, args.num_samples, visualizer, adv_vis, device, args
        )
    
    elif args.mode == 'compare':
        # 比较模式 - 需要至少两个案例ID
        if not args.case_ids or len(args.case_ids) < 2:
            print("Compare mode requires at least two case IDs")
            return
        
        # 查找案例索引
        case_indices = []
        for case_id in args.case_ids:
            if case_id in dataset.name_list:
                case_indices.append(dataset.name_list.index(case_id))
            else:
                print(f"Case {case_id} not found in dataset")
        
        if len(case_indices) < 2:
            print("Not enough valid cases for comparison")
            return
        
        # 运行每个样本的可视化
        results = []
        for case_idx in case_indices:
            result = run_single_case_visualization(
                model, dataset, case_idx, visualizer, adv_vis, device, args
            )
            results.append(result)
    
    # 创建汇总报告
    if len(results) > 0:
        report_path = create_summary_report(results, args.output_dir)
    
    print("Visualization completed!")


if __name__ == "__main__":
    main()
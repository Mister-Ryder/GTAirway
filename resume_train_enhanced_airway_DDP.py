'''
@author: pdl贾岩
@file: resume_train_enhanced_airway_DDP_gau.py
@time: 2025/04/09
@intro:
        该版本是train的resume版本，以实现从最后epoch 参数加载同日志下接续训练
'''


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from func.load_dataset import airway_dataset, PKLtoNPYAdapter, ResetableRandomSampler, seed_everything, ClassBalancedSampler,DistributedClassBalancedSampler
from func.model_arch_model10 import SegAirwayModel
from func.AnatomicalClassifier import AirwayAnatomicalClassifier
from func.EnhancedAirwaySegmentation2 import EnhancedAirwaySegmentation
from func.loss_func import dice_loss_weights, dice_accuracy,dice_loss_power_weights
import os


# 在main函数之前添加此代码
class Args:
    """参数类，存储所有训练配置"""
    def __init__(self):
        # 训练参数
        self.model_name = "enhanced_airway_segmentation"
        self.accumulated_steps=5 #累积次数
        self.learning_rate = 5e-5 * self.accumulated_steps
        self.max_epoch = 200
        self.num_samples_of_each_epoch = 2500
        self.batch_size = 6
        self.train_file_format = '.npy'
        self.crop_size = (128, 128, 128)
        self.windowMin_CT_img_HU = -1000
        self.windowMax_CT_img_HU = 600
        self.model_save_freq = 6
        self.num_workers = 1  # 每个GPU进程使用的工作线程数减少
        self.prefetch_factor = 1
        self.memory_cleanup_freq = 1
        self.seed = 1419

        # 模型路径
        self.segmentation_model_path = '/home/ubuntu/ICME/checkpoint/BAS_model10-200epoch_-ronghe-xiuzheng_599.pkl'
        self.classifier_model_path = '/home/ubuntu/ICME/runs_anatomical_classifier/20250402-073130_improved/airway_anatomical_classifier_improved_best.pkl'

        # 数据路径
        self.pkl_path = "./dataset_info/airway_metadata_train_anatomical.pkl"
        self.npy_base_dir = "/home/ubuntu/shared-data/Precrop_dataset/Precrop_dataset_for_BAS"
        # self.cache_dir = "/home/ubuntu/shared-data/BAS/train/.airway_cache"

        # 添加恢复训练相关参数
        self.resume = False  # 是否从检查点恢复训练
        self.resume_checkpoint = None  # 检查点路径
        self.resume_log_dir = None  # 恢复训练时使用的日志目录


# 修改main函数，添加分布式训练支持
def main_worker(gpu, ngpus_per_node, args):
    """每个GPU进程的主要工作函数"""
    args.gpu = gpu
    
    # 设置当前设备
    torch.cuda.set_device(args.gpu)
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:23456',
        world_size=ngpus_per_node,
        rank=args.gpu
    )

    # 创建日志目录(只在主进程)
    if args.gpu == 0:
        if args.resume and args.resume_log_dir:
            # 如果从检查点恢复训练，使用指定的日志目录
            log_dir = args.resume_log_dir
            print(f"恢复训练使用日志目录: {log_dir}")
        else:
            # 否则创建新的日志目录
            current_time = time.strftime('%Y%m%d-%H%M%S')
            log_dir = os.path.join('runs_enhanced_airway', current_time)
            os.makedirs(log_dir, exist_ok=True)
            print(f"创建新的日志目录: {log_dir}")

        writer = SummaryWriter(log_dir=log_dir)
    else:
        log_dir = ""
        writer = None

    # 初始化增强型分割模型 - 直接加载预训练权重
    model = EnhancedAirwaySegmentation(
        segmentation_model_path=args.segmentation_model_path,
        classifier_model_path=args.classifier_model_path
    )
    
    # 将模型移到GPU
    model.cuda(args.gpu)
    
    # 使用DDP包装模型
    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    
    # # 优化器 - 只优化分割网络和融合模块参数
    # trainable_params = [p for name, p in model.named_parameters()
    #                   if not name.startswith('module.classifier') and p.requires_grad]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # # # 学习率调度器
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)

    # 初始化训练状态变量
    start_epoch = 0
    best_dice = 0.0
    best_loss = float('inf')
    global_step = 0
    # 多阶段训练策略
    II_start = 50  # 半联合训练
    III_start = 70  # 联合训练
    joint_training = False

    # 从检查点恢复训练
    if args.resume and args.resume_checkpoint:
        if args.gpu == 0:
            print(f"正在从检查点恢复训练: {args.resume_checkpoint}")
        # 加载检查点
        checkpoint = torch.load(args.resume_checkpoint, map_location=f'cuda:{args.gpu}')
        # 恢复模型状态
        model.module.load_state_dict(checkpoint['model_state_dict'])
        # 恢复训练状态
        start_epoch = checkpoint['epoch'] + 1
        if start_epoch >= II_start and start_epoch < III_start:
            if args.gpu == 0:
                print("第2阶段: 解冻分类器的深层特征")
            if hasattr(model, 'module'):
                model.module.unfreeze_classifier_deep_layers()
            else:
                model.unfreeze_classifier_deep_layers()
            # 更新优化器
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate * 0.5, weight_decay=1e-4)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - II_start, eta_min=1e-6)
        elif start_epoch >= III_start:
            if args.gpu == 0:
                print("第3阶段: 全面联合微调")
            if hasattr(model, 'module'):
                model.module.unfreeze_all_classifier()
            else:
                model.unfreeze_all_classifier()
            # 更新优化器
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate * 0.1, weight_decay=1e-4)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - III_start, eta_min=1e-7)
            # 启用联合训练模式
            joint_training = True
        # 恢复优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'best_dice' in checkpoint:
            best_dice = checkpoint['best_dice']
        else:
            best_dice = checkpoint['dice']
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
        else:
            best_loss = checkpoint['loss']
        # 计算全局步数 (对TensorBoard很重要)
        # 假设每个epoch有相同数量的步数
        global_step = start_epoch * (args.num_samples_of_each_epoch // (args.batch_size * ngpus_per_node))
        if args.gpu == 0:
            print(f"成功恢复训练，从第 {start_epoch} 个epoch开始，最佳dice={best_dice:.4f}，最佳loss={best_loss:.4f}")
    
    # 数据加载
    adapter = PKLtoNPYAdapter(args.pkl_path, args.npy_base_dir)
    data_dict = adapter.get_adapted_dict()
    
    # # 创建缓存目录
    # if args.gpu == 0:
    #     os.makedirs(args.cache_dir, exist_ok=True)
    
    # 创建数据集
    train_dataset = airway_dataset(data_dict,
                                 add_anatomical_features=True, model_type="full")
    train_dataset.set_para(
        file_format=args.train_file_format,
        crop_size=args.crop_size,
        windowMin=args.windowMin_CT_img_HU,
        windowMax=args.windowMax_CT_img_HU,
        need_tensor_output=True,
        need_transform=True
    )
    
    # DDP需要使用DistributedSampler替代普通采样器
    train_sampler = DistributedClassBalancedSampler(
        train_dataset,
        num_replicas=ngpus_per_node,
        rank=args.gpu,
        class_key='generation_type',
        num_samples=min(args.num_samples_of_each_epoch, len(data_dict)),
        replacement=True,
        seed=args.seed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
        sampler=train_sampler
    )
    
    # 训练循环
    start_time = time.time()
    
    for epoch in range(start_epoch, args.max_epoch):
    # 训练循环从保存点继续
        # 设置采样器的epoch
        train_sampler.set_epoch(epoch)

        if epoch == II_start:
            if args.gpu == 0:
                print("第2阶段: 解冻分类器的深层特征")
            # # 解冻分类器的后半部分
            # for i, param in enumerate(model.module.classifier.parameters()):
            #     if i >= len(list(model.module.classifier.parameters())) // 2:
            #         param.requires_grad = True
            if hasattr(model, 'module'):
                model.module.unfreeze_classifier_deep_layers()
            else:
                model.unfreeze_classifier_deep_layers()

            # 更新优化器
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate * 0.5, weight_decay=1e-4)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - II_start, eta_min=1e-6)
        
        elif epoch == III_start:
            if args.gpu == 0:
                print("第3阶段: 全面联合微调")
            # # 解冻所有分类器参数
            # for param in model.module.classifier.parameters():
            #     param.requires_grad = True
            if hasattr(model, 'module'):
                model.module.unfreeze_all_classifier()
            else:
                model.unfreeze_all_classifier()
            
            # 更新优化器
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate * 0.1, weight_decay=1e-4)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch - III_start, eta_min=1e-7)
            
            # 启用联合训练模式
            joint_training = True
        
        # 训练模式
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_cls_loss = 0.0

        # 清零梯度
        optimizer.zero_grad()

        
        for batch_idx, batch in enumerate(train_loader):

            # 获取输入和目标
            inputs = batch['image'].float().cuda(args.gpu, non_blocking=True)
            labels = batch['label'].float().cuda(args.gpu, non_blocking=True)
            gen_type = batch['generation_type'].long().cuda(args.gpu, non_blocking=True)
            struct_type = batch['structure_type'].long().cuda(args.gpu, non_blocking=True)
            loc_type = batch['location_type'].long().cuda(args.gpu, non_blocking=True)


            # 前向传播
            segmentation ,cls_output = model(inputs, joint_training=joint_training)

            # 计算损失 - 使用增强的解剖学加权损失, cls_output
            with torch.no_grad():
                # 1. 代数加权 - 高代数给更高权重
                gen_weights = torch.ones_like(labels)
                for i in range(7):
                    if i >= 5:  # 细小气道
                        weight_factor = 1.5
                    elif i >= 3:  # 段支气管
                        weight_factor = 1.2
                    else:  # 主干气道
                        weight_factor = 1.0

                    gen_mask = (gen_type == i).view(-1, 1, 1, 1, 1).float()
                    gen_weights = gen_weights + gen_mask * (weight_factor - 1.0)

                # 2. 结构加权 - 分叉区域和直管道加权
                struct_weights = torch.ones_like(labels)
                bifurcation_mask = (struct_type == 3).view(-1, 1, 1, 1, 1).float()
                struct_weights = struct_weights + bifurcation_mask * 0.3

                # 3. 组合权重
                combined_weights = gen_weights * struct_weights

            # 4. 计算带权重的Dice损失
            seg_loss = dice_loss_weights(segmentation[:, 0, :, :, :], 1.0 - labels, combined_weights) + \
                     dice_loss_power_weights(segmentation[:, 1, :, :, :], labels, combined_weights, alpha=2)

            # 5. 联合训练时添加分类器损失
            if joint_training and 'enhanced_features' in cls_output:
                # Focal Loss改善结构分类
                struct_loss = focal_loss(cls_output['structure'], struct_type, alpha=0.5, gamma=2.5)

                # 标准交叉熵损失用于代数和位置
                gen_loss = F.cross_entropy(cls_output['generation'], gen_type)
                loc_loss = F.cross_entropy(cls_output['location'], loc_type)

                # 对于结构不确定性估计的监督（如果有）
                cls_uncertainty_loss = 0.0
                if 'uncertainty_estimates' in cls_output:
                    # 获取不确定性估计
                    struct_uncertainty = cls_output['uncertainty_estimates']['structure']

                    # 计算真实不确定性作为目标
                    struct_pred = torch.argmax(cls_output['structure'], dim=1).detach()
                    struct_error = (struct_pred != struct_type).float().detach()
                    struct_certainty_target = 1.0 - struct_error  # 转换为确定性目标

                    if len(struct_uncertainty.shape) > 2:
                        # Reshape to [batch_size, num_classes]
                        struct_uncertainty = struct_uncertainty.view(struct_uncertainty.size(0),
                                                                     struct_uncertainty.size(1))

                    # Get the model's certainty prediction
                    struct_uncertainty_probs = F.softmax(struct_uncertainty, dim=1)

                    # Get probabilities for the correct class for each sample
                    batch_indices = torch.arange(struct_type.size(0), device=struct_type.device)
                    struct_certainty_pred = struct_uncertainty_probs[batch_indices, struct_type]
                    cls_uncertainty_loss = F.mse_loss(struct_certainty_pred, struct_certainty_target)

                # 结合所有分类损失 - 结构损失权重更高
                cls_loss = gen_loss + 1.5 * struct_loss + loc_loss + 0.5 * cls_uncertainty_loss

                # 将分类损失添加到总损失中
                cls_weight = min(0.5, (epoch - III_start) * 0.01 + 0.1)  # 从0.1开始，最大0.5
                total_loss = seg_loss + cls_weight * cls_loss

                # 同步clc损失值
                cls_loss_tensor = torch.tensor([cls_loss.item()]).cuda(args.gpu)
                dist.all_reduce(cls_loss_tensor, op=dist.ReduceOp.SUM)
                running_cls_loss += cls_loss_tensor.item() / ngpus_per_node
                # running_cls_loss += cls_loss.item()
            else:
                total_loss = seg_loss

            # 6. 添加额外监督信号（可选）
            if 'guidance_map' in cls_output and epoch >= 40:
                guidance_map = cls_output['guidance_map']
                guidance_target = F.avg_pool3d(labels, kernel_size=4, stride=4).detach()
                if guidance_map.shape[2:] != guidance_target.shape[2:]:
                    guidance_map = F.interpolate(guidance_map, size=guidance_target.shape[2:],
                                               mode='trilinear', align_corners=False)

                guidance_loss = F.mse_loss(guidance_map, guidance_target)
                total_loss = total_loss + 0.1 * guidance_loss

            total_loss = total_loss / args.accumulated_steps

            # 反向传播
            total_loss.mean().backward()
            # # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # optimizer.step()
            # optimizer.zero_grad()
            if (batch_idx + 1) % args.accumulated_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                # 统计
                # 同步损失值
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                running_loss += total_loss.item()/ngpus_per_node
                # 同步dice指标
                dice_val = dice_accuracy(segmentation[:, 1, :, :, :], labels)
                # 在计算dice_val后立即同步
                dice_val_tensor = torch.tensor([dice_val]).cuda(args.gpu)
                dist.all_reduce(dice_val_tensor, op=dist.ReduceOp.SUM)
                dice_val = dice_val_tensor.item() / ngpus_per_node
                running_dice += dice_val

                # 仅在主进程打印
                if args.gpu == 0 and (batch_idx + 1) % 5 == 0:
                    elapsed = time.time() - start_time
                    avg_loss = running_loss / (batch_idx + 1)
                    avg_dice = running_dice / (batch_idx + 1)

                    progress_info = (
                        f"Epoch: [{epoch + 1}/{args.max_epoch}] "
                        f"Batch: [{batch_idx + 1}/{len(train_loader)}] "
                        f"Loss: {avg_loss:.4f} "
                        f"Dice: {avg_dice:.4f} "
                        f"Current Dice: {dice_val:.4f} "
                    )

                    if joint_training:
                        avg_cls_loss = running_cls_loss / (batch_idx + 1)
                        progress_info += f"Cls Loss: {avg_cls_loss:.4f} "

                    progress_info += f"Time: {elapsed:.2f}s"
                    print(progress_info)

                    # 添加到TensorBoard
                    if writer is not None:
                        step = global_step + batch_idx
                        writer.add_scalar('Loss/train', avg_loss, step)
                        writer.add_scalar('Dice/train', avg_dice, step)
                        if joint_training:

                            writer.add_scalar('Loss/classifier', avg_cls_loss, step)

            # # 定期清理显存
            # if (batch_idx + 1) % args.memory_cleanup_freq == 0:
            #     torch.cuda.empty_cache()
            # # 清理内存
            # del inputs, labels, gen_type, struct_type, loc_type
            # del segmentation, cls_output, total_loss
            # if 'gen_weights' in locals(): del gen_weights
            # if 'struct_weights' in locals(): del struct_weights
            # if 'combined_weights' in locals(): del combined_weights
            # if 'seg_loss' in locals(): del seg_loss
            # import gc
            # gc.collect()
            # torch.cuda.empty_cache()

        # 更新全局步数
        global_step += len(train_loader)
        
        # # 更新学习率
        # scheduler.step()
        with torch.no_grad():
            # 计算epoch统计信息 - 在多GPU上同步
            avg_loss = running_loss / len(train_loader)
            avg_dice = running_dice / len(train_loader)
            avg_cls_loss = running_cls_loss / len(train_loader) if joint_training else 0.0

            # # 收集所有GPU的指标
            # if ngpus_per_node > 1:
            #     avg_loss_tensor = torch.tensor([avg_loss]).cuda(args.gpu)
            #     avg_dice_tensor = torch.tensor([avg_dice]).cuda(args.gpu)
            #
            #     # 在所有进程上平均指标
            #     dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            #     dist.all_reduce(avg_dice_tensor, op=dist.ReduceOp.SUM)
            #
            #     avg_loss = avg_loss_tensor.item() / ngpus_per_node
            #     avg_dice = avg_dice_tensor.item() / ngpus_per_node
            #
            #     if joint_training:
            #         avg_cls_loss_tensor = torch.tensor([avg_cls_loss]).cuda(args.gpu)
            #         dist.all_reduce(avg_cls_loss_tensor, op=dist.ReduceOp.SUM)
            #         avg_cls_loss = avg_cls_loss_tensor.item() / ngpus_per_node
        
        # 仅在主进程打印和保存
        if args.gpu == 0:
            # 打印epoch结果
            epoch_info = (
                f"Epoch: {epoch + 1}/{args.max_epoch} "
                f"Loss: {avg_loss:.4f} "
                f"Dice: {avg_dice:.4f} "
            )
            
            if joint_training:
                epoch_info += f"Cls Loss: {avg_cls_loss:.4f} "
                
            epoch_info += f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            print(epoch_info)
            
            # 添加到TensorBoard
            if writer is not None:
                writer.add_scalar('Epoch/loss', avg_loss, epoch)
                writer.add_scalar('Epoch/dice', avg_dice, epoch)
                if joint_training:
                    writer.add_scalar('Epoch/cls_loss', avg_cls_loss, epoch)
                writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 保存模型 - 只需保存模块状态而非整个DDP包装器
            if (epoch + 1) % args.model_save_freq == 0:
                model_path = os.path.join(log_dir, f"{args.model_name}_epoch{epoch + 1}.pkl")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # 注意这里保存.module的状态
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'dice': avg_dice,
                    'best_dice': best_dice,
                    'best_loss': best_loss,
                    'global_step': global_step
                }, model_path)
                print(f"模型已保存到 {model_path}")
            
            # 保存最佳模型 (基于Dice)
            if avg_dice > best_dice:
                best_dice = avg_dice
                best_dice_model_path = os.path.join(log_dir, f"{args.model_name}_best_dice.pkl")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # 保存.module的状态
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'dice': best_dice,
                    'best_dice': best_dice,
                    'best_loss': best_loss,
                    'global_step': global_step
                }, best_dice_model_path)
                print(f"最佳Dice模型已保存到 {best_dice_model_path}")
            
            # 保存最佳模型 (基于Loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_loss_model_path = os.path.join(log_dir, f"{args.model_name}_best_loss.pkl")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),  # 保存.module的状态
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'dice': avg_dice,
                    'best_dice': best_dice,
                    'best_loss': best_loss,
                    'global_step': global_step
                }, best_loss_model_path)
                print(f"最佳Loss模型已保存到 {best_loss_model_path}")
        
        # 每个epoch结束后清理资源
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # 关闭SummaryWriter
    if writer is not None:
        writer.close()
    if args.gpu == 0:
        print("训练完成!")

# 添加结构损失的Focal Loss
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    pred_prob = F.softmax(pred, dim=1)
    p_t = pred_prob.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = -alpha * (1 - p_t) ** gamma * torch.log(p_t + 1e-10)
    return loss.mean()




def main():
    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDNN_ENABLED'] = '1'
    os.environ['CUDNN_BENCHMARK'] = '0'
    
    # 设置随机种子
    seed = 1419
    seed_everything(seed)

    # 创建参数对象 - 现在使用全局定义的Args类
    args = Args()
    args.seed = seed  # 确保使用相同的种子

    # 添加命令行参数解析
    #python resume_train_enhanced_airway_DDP.py --resume --resume_checkpoint /home/ubuntu/ICME/runs_enhanced_airway/20250408-121033/enhanced_airway_segmentation_epoch66.pkl --resume_log_dir /home/ubuntu/ICME/runs_enhanced_airway/20250408-121033

    import argparse
    parser = argparse.ArgumentParser(description='Enhanced Airway Segmentation Training')
    parser.add_argument('--resume', action='store_true', help='是否从检查点恢复训练')
    parser.add_argument('--resume_checkpoint', type=str, help='恢复训练使用的检查点路径')
    parser.add_argument('--resume_log_dir', type=str, help='恢复训练使用的日志目录')
    cmd_args = parser.parse_args()
    # 将命令行参数传递给args对象
    if cmd_args.resume:
        args.resume = True
        args.resume_checkpoint = cmd_args.resume_checkpoint
        args.resume_log_dir = cmd_args.resume_log_dir

    
    # 检测GPU数量
    ngpus_per_node = torch.cuda.device_count()
    
    # 多进程启动
    if ngpus_per_node > 1:
        print(f"使用 {ngpus_per_node} 个GPU进行分布式训练")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # 单GPU模式
        print("使用单GPU训练")
        main_worker(0, 1, args)

if __name__ == "__main__":
    main()

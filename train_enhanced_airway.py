# train_enhanced_airway.py - 增强版
import os

# # 设置环境变量，允许 PyTorch 使用扩展段内存分配器，减少碎片
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# # 禁用某些可能导致cuDNN错误的优化
# os.environ['CUDNN_ENABLED'] = '1'
# os.environ['CUDNN_BENCHMARK'] = '0'

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from func.load_dataset import airway_dataset, PKLtoNPYAdapter, ResetableRandomSampler, seed_everything, ClassBalancedSampler,RandomSampler
from func.model_arch_model10 import SegAirwayModel
from func.AnatomicalClassifier import AirwayAnatomicalClassifier
from func.EnhancedAirwaySegmentation2 import EnhancedAirwaySegmentation
from func.loss_func import dice_loss_weights, dice_accuracy,dice_loss_power_weights

import gc



# 设置随机种子
seed = 1419
seed_everything(seed)

# 训练参数
current_time = time.strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('runs_enhanced_airway', current_time)
writer = SummaryWriter(log_dir=log_dir)

model_name = "enhanced_airway_segmentation"
learning_rate = 5e-5
max_epoch = 200
num_samples_of_each_epoch = 2500
batch_size = 4
train_file_format = '.npy'
crop_size = (128, 128, 128)
windowMin_CT_img_HU = -1000
windowMax_CT_img_HU = 600
model_save_freq = 5
num_workers = 1
# prefetch_factor = 4

# 添加内存清理频率，防止显存碎片化
memory_cleanup_freq = 1  # 每处理多少个batch清理一次内存

# 模型路径
segmentation_model_path = '/home/ubuntu/ICME/checkpoint/BAS_model10-200epoch_-ronghe-xiuzheng_599.pkl'
classifier_model_path = '/home/ubuntu/ICME/runs_anatomical_classifier/20250402-073130_improved/airway_anatomical_classifier_improved_best.pkl'

# 数据路径
pkl_path = "./dataset_info/airway_metadata_train_anatomical.pkl"
npy_base_dir = "/home/ubuntu/shared-data/Precrop_dataset/Precrop_dataset_for_BAS"
# # 初始化基础模型
# base_model = SegAirwayModel(in_channels=1, out_channels=2)
#
# # 初始化分类器
# classifier = AirwayAnatomicalClassifier(
#     in_channels=1,
#     generation_classes=7,
#     structure_classes=7,
#     location_classes=7
# )

# # 加载预训练权重
# if os.path.exists(segmentation_model_path):
#     print(f"加载分割模型预训练权重: {segmentation_model_path}")
#     checkpoint = torch.load(segmentation_model_path, map_location='cpu')
#     state_dict = checkpoint['model_state_dict']
#     # 移除"module."前缀（如果有）
#     new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     base_model.load_state_dict(new_state_dict, strict=False)
#
# if os.path.exists(classifier_model_path):
#     print(f"加载分类器预训练权重: {classifier_model_path}")
#     checkpoint = torch.load(classifier_model_path, map_location='cpu')
#     state_dict = checkpoint['model_state_dict']
#     # 移除"module."前缀（如果有）
#     new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     classifier.load_state_dict(new_state_dict, strict=False)

# 创建增强型分割模型
# model = EnhancedAirwaySegmentation(base_model, classifier)

# 初始化增强型分割模型 - 直接加载预训练权重
model = EnhancedAirwaySegmentation(
    segmentation_model_path=segmentation_model_path,
    classifier_model_path=classifier_model_path
)

# 多GPU支持
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU!")
    # # 分配模型的不同部分到不同的GPU
    # model.base_model = model.base_model.to('cuda:1')  # 基础分割模型放在GPU 1
    # model.classifier = model.classifier.to('cuda:0')  # 分类器放在GPU 0

    # 然后用DataParallel包装，但指定GPU 1为主设备
    # model = nn.DataParallel(model, device_ids=[0, 1], output_device=1)
    # 确保模型初始参数也在正确的设备上
    # model = model.to('cuda:1')
    model = nn.DataParallel(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# # 优化器 - 只优化分割网络和融合模块参数
# # 过滤出分类器参数，不包含在优化中
# trainable_params = [p for name, p in model.named_parameters()
#                     if not name.startswith('classifier') and p.requires_grad]
#
# optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
#
# # 学习率调度器
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

# 优化器 - 只训练非分类器参数
# trainable_params = [p for name, p in model.named_parameters()
#                    if not name.split('.')[1 if 'module' in name else 0] == 'classifier'
#                    and p.requires_grad]

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)
# del trainable_params
# gc.collect(trainable_params)
# 数据加载
adapter = PKLtoNPYAdapter(pkl_path, npy_base_dir)
data_dict = adapter.get_adapted_dict()

# 创建缓存目录
cache_dir = "/home/ubuntu/shared-data/BAS/train/.airway_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# 创建数据集 - 完整模式
train_dataset = airway_dataset(data_dict, cache_dir=cache_dir,
                               add_anatomical_features=True, model_type="full")
train_dataset.set_para(
    file_format=train_file_format,
    crop_size=crop_size,
    windowMin=windowMin_CT_img_HU,
    windowMax=windowMax_CT_img_HU,
    need_tensor_output=True,
    need_transform=True
)

# 创建数据加载器 - 使用平衡采样器提高训练效果
sampler = ClassBalancedSampler(
    train_dataset,
    num_samples=min(num_samples_of_each_epoch, len(data_dict)),
    class_key='generation_type',  # 优先平衡代数类型
    replacement=True
)
# sampler = RandomSampler(
#             train_dataset,
#             num_samples=min(num_samples_of_each_epoch, len(data_dict)),
#             replacement=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    # prefetch_factor=prefetch_factor,
    sampler=sampler
)

# 添加结构损失的Focal Loss
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    pred_prob = F.softmax(pred, dim=1)
    p_t = pred_prob.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = -alpha * (1 - p_t) ** gamma * torch.log(p_t + 1e-10)
    return loss.mean()

# 训练循环
start_time = time.time()
best_dice = 0.0
best_loss = float('inf')

# 三阶段训练策略
for epoch in range(max_epoch):

    # 多阶段训练策略
    # 第1阶段 (1-50 epochs): 在分类器冻结的情况下训练
    # 第2阶段 (51-100 epochs): 解冻分类器的深层特征
    # 第3阶段 (101+ epochs): 全面联合微调
    joint_training = False

    if epoch == 40:
        print("第2阶段: 解冻分类器的深层特征")
        batch_size=6
        # memory_cleanup_freq = 1  # 每处理多少个batch清理一次内存
        # # 解冻分类器的后半部分
        # for i, param in enumerate(model.module.classifier.parameters() if isinstance(model, nn.DataParallel)
        #                           else model.classifier.parameters()):
        #     if i >= len(list(model.module.classifier.parameters() if isinstance(model, nn.DataParallel)
        #                      else model.classifier.parameters())) // 2:
        #         param.requires_grad = True
        # del optimizer
        # del scheduler
        # torch.cuda.empty_cache()  # 清理CUDA缓存
        if isinstance(model, nn.DataParallel):
            model.module.unfreeze_classifier_deep_layers()
        else:
            model.unfreeze_classifier_deep_layers()
        # 更新优化器以包含新解冻参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate * 0.5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch - 40, eta_min=1e-6)

    elif epoch == 70:
        print("第3阶段: 全面联合微调")
        batch_size = 6
        # memory_cleanup_freq = 1  # 每处理多少个batch清理一次内存
        # # 解冻所有分类器参数
        # for param in model.module.classifier.parameters() if isinstance(model,
        #                                                                 nn.DataParallel) else model.classifier.parameters():
        #     param.requires_grad = True
        # del optimizer
        # del scheduler
        # torch.cuda.empty_cache()  # 清理CUDA缓存
        if isinstance(model, nn.DataParallel):
            model.module.unfreeze_all_classifier()
        else:
            model.unfreeze_all_classifier()
        # 更新优化器
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate * 0.1, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch - 70, eta_min=1e-7)

        # 启用联合训练模式
        joint_training = True

    # 设置采样器的epoch
    if hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch)

    # 重新创建DataLoader保证资源释放
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        # prefetch_factor=prefetch_factor,
        sampler=sampler
    )


    # 训练模式
    # model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_cls_loss = 0.0  # 用于联合训练

    for batch_idx, batch in enumerate(train_loader):

        # 获取输入和目标
        inputs = batch['image'].float().to(device, non_blocking=True)
        labels = batch['label'].float().to(device, non_blocking=True)
        gen_type = batch['generation_type'].long().to(device, non_blocking=True)
        struct_type = batch['structure_type'].long().to(device, non_blocking=True)
        loc_type = batch['location_type'].long().to(device, non_blocking=True)#, non_blocking=True

        # 清零梯度
        optimizer.zero_grad(set_to_none=True)  # 替换原有的 zero_grad()

        # 前向传播
        segmentation, cls_output = model(inputs, joint_training=joint_training)


        # 计算损失 - 使用增强的解剖学加权损失
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
            del gen_mask
            # 2. 结构加权 - 分叉区域和直管道加权
            struct_weights = torch.ones_like(labels)
            bifurcation_mask = (struct_type == 3).view(-1, 1, 1, 1, 1).float()
            # tube_mask = (struct_type == 1).view(-1, 1, 1, 1, 1).float()  # 为直管道添加额外权重
            struct_weights = struct_weights + bifurcation_mask * 0.3 #+ tube_mask * 0.4  # 直管道加40%权重
            del bifurcation_mask
            # 3. 组合权重
            combined_weights = gen_weights * struct_weights

        # 4. 计算带权重的Dice损失
        # seg_loss = dice_loss_weights(segmentation, labels, weights=combined_weights)
        # 1. 修改train_enhanced_airway.py中的dice_loss_weights调用方式
        seg_loss = dice_loss_weights(segmentation[:, 0, :, :, :], 1.0 - labels, combined_weights) + \
                   dice_loss_power_weights(segmentation[:, 1, :, :, :], labels, combined_weights, alpha=2)

        # 5. 联合训练时添加分类器损失
        if joint_training and 'enhanced_features' in cls_output:
            # 使用Focal Loss改善结构分类
            struct_loss = focal_loss(cls_output['structure'], struct_type, alpha=0.5, gamma=2.5)

            # 标准交叉熵损失用于代数和位置
            gen_loss = F.cross_entropy(cls_output['generation'], gen_type)
            loc_loss = F.cross_entropy(cls_output['location'], loc_type)

            # 对于结构不确定性估计的监督（如果有）
            cls_uncertainty_loss = 0.0
            if 'uncertainty_estimates' in cls_output:
                # # 获取不确定性估计
                # struct_uncertainty = cls_output['uncertainty_estimates']['structure']
                #
                # # 计算真实不确定性作为目标
                # # 例如，对于错误分类的样本，不确定性应该更高
                # struct_pred = torch.argmax(cls_output['structure'], dim=1)
                # struct_error = (struct_pred != struct_type).float().view(-1, 1, 1, 1)
                # struct_certainty_target = 1.0 - struct_error  # 转换为确定性目标
                #
                # # 计算不确定性损失 - 使用MSE
                # uncertainty_target = F.adaptive_avg_pool3d(struct_certainty_target, output_size=1)
                # struct_uncertainty_probs = F.softmax(struct_uncertainty, dim=1)
                # struct_certainty_pred = struct_uncertainty_probs[:, struct_type.view(-1)]
                # cls_uncertainty_loss = F.mse_loss(struct_certainty_pred, uncertainty_target.squeeze())
                # Replace the uncertainty loss calculation block (around line 285) with this:

                # Get uncertainty estimates
                struct_uncertainty = cls_output['uncertainty_estimates']['structure']

                # Calculate ground truth uncertainty (based on prediction error)
                struct_pred = torch.argmax(cls_output['structure'], dim=1)  # Get predictions
                struct_error = (struct_pred != struct_type).float()  # Binary error: 1 if wrong, 0 if correct
                struct_certainty_target = 1.0 - struct_error  # Convert to certainty: 1 if correct, 0 if wrong

                # Ensure uncertainty has the proper shape before softmax
                # If struct_uncertainty has shape [8, 8, 1, 1, 1], we need to reshape it
                if len(struct_uncertainty.shape) > 2:
                    # Reshape to [batch_size, num_classes]
                    struct_uncertainty = struct_uncertainty.view(struct_uncertainty.size(0),
                                                                 struct_uncertainty.size(1))

                # Get the model's certainty prediction
                struct_uncertainty_probs = F.softmax(struct_uncertainty, dim=1)

                # Get probabilities for the correct class for each sample
                batch_indices = torch.arange(struct_type.size(0), device=struct_type.device)
                struct_certainty_pred = struct_uncertainty_probs[batch_indices, struct_type]

                # Now both tensors should have shape [batch_size]
                # Calculate MSE loss with matching shapes
                cls_uncertainty_loss = F.mse_loss(struct_certainty_pred, struct_certainty_target)
                print("cls_uncertainty_loss", cls_uncertainty_loss)

            # 结合所有分类损失 - 结构损失权重更高
            cls_loss = gen_loss + 1.5 * struct_loss + loc_loss + 0.5 * cls_uncertainty_loss

            # 将分类损失添加到总损失中，使用逐渐增加的权重
            cls_weight = min(0.5, (epoch - 100) * 0.01 + 0.1)  # 从0.1开始，最大0.5
            total_loss = seg_loss + cls_weight * cls_loss
            running_cls_loss += cls_loss.item()
        else:
            total_loss = seg_loss

        # 6. 添加额外监督信号（可选）
        # 如果使用了分类器的解剖学引导图，添加额外监督
        if 'guidance_map' in cls_output and epoch >= 40:  # 第2阶段开始添加
            guidance_map = cls_output['guidance_map']
            # 创建简化的引导标签（主要气道轮廓）
            guidance_target = F.avg_pool3d(labels, kernel_size=4, stride=4)
            if guidance_map.shape[2:] != guidance_target.shape[2:]:
                guidance_map = F.interpolate(guidance_map, size=guidance_target.shape[2:],
                                             mode='trilinear', align_corners=False)

            guidance_loss = F.mse_loss(guidance_map, guidance_target)
            total_loss = total_loss + 0.1 * guidance_loss  # 权重较小

        # 反向传播
        total_loss.backward()

        # # 梯度裁剪，防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 统计
        running_loss += total_loss.item()
        dice_val =  dice_accuracy(segmentation[:, 1, :, :, :], labels)
        running_dice += dice_val

        # 打印进度
        if (batch_idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / (batch_idx + 1)
            avg_dice = running_dice / (batch_idx + 1)

            progress_info = (
                f"Epoch: [{epoch + 1}/{max_epoch}] "
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
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train', avg_loss, step)
            writer.add_scalar('Dice/train', avg_dice, step)
            if joint_training:
                writer.add_scalar('Loss/classifier', avg_cls_loss, step)

        # del inputs, labels, gen_type, struct_type, loc_type
        # del segmentation, cls_output, total_loss
        # del gen_weights,struct_weights,combined_weights
        # if 'gen_weights' in locals(): del gen_weights
        # if 'struct_weights' in locals(): del struct_weights
        # if 'combined_weights' in locals(): del combined_weights
        # if 'seg_loss' in locals(): del seg_loss
        # if 'cls_loss' in locals(): del cls_loss
        # if 'guidance_map' in locals(): del guidance_map
        # if 'guidance_target' in locals(): del guidance_target
        # # 强制垃圾回收和清理显存
        gc.collect()
        torch.cuda.empty_cache()
        # 定期清理显存，防止碎片化
        # if (batch_idx + 1) % memory_cleanup_freq == 0:
        #     torch.cuda.empty_cache()

    # # 更新学习率
    # scheduler.step()

    # 计算epoch统计信息
    avg_loss = running_loss / len(train_loader)
    avg_dice = running_dice / len(train_loader)
    avg_cls_loss = running_cls_loss / len(train_loader) if joint_training else 0.0

    # 打印epoch结果
    epoch_info = (
        f"Epoch: {epoch + 1}/{max_epoch} "
        f"Loss: {avg_loss:.4f} "
        f"Dice: {avg_dice:.4f} "
    )

    if joint_training:
        epoch_info += f"Cls Loss: {avg_cls_loss:.4f} "

    epoch_info += f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    print(epoch_info)

    # 添加到TensorBoard
    writer.add_scalar('Epoch/loss', avg_loss, epoch)
    writer.add_scalar('Epoch/dice', avg_dice, epoch)
    if joint_training:
        writer.add_scalar('Epoch/cls_loss', avg_cls_loss, epoch)
    writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)

    # 保存模型
    if (epoch + 1) % model_save_freq == 0:
        model_path = os.path.join(log_dir, f"{model_name}_epoch{epoch + 1}.pkl")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'dice': avg_dice
        }, model_path)
        print(f"模型已保存到 {model_path}")

    # 保存最佳模型 (基于Dice)
    if avg_dice > best_dice:
        best_dice = avg_dice
        best_dice_model_path = os.path.join(log_dir, f"{model_name}_best_dice.pkl")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'dice': best_dice
        }, best_dice_model_path)
        print(f"最佳Dice模型已保存到 {best_dice_model_path}")

    # 保存最佳模型 (基于Loss)
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_loss_model_path = os.path.join(log_dir, f"{model_name}_best_loss.pkl")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
            'dice': avg_dice
        }, best_loss_model_path)
        print(f"最佳Loss模型已保存到 {best_loss_model_path}")

    # 每个epoch结束后清理资源
    del train_loader
    # 在重建 DataLoader 前添加
    if 'train_loader' in locals():
        del train_loader

    gc.collect()  # 主动触发垃圾回收
    # # 每个epoch结束后清理缓存
    torch.cuda.empty_cache()

# 关闭SummaryWriter
writer.close()
print("训练完成!")

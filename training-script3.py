# train
from func.load_dataset import airway_dataset, PKLtoNPYAdapter,ResetableRandomSampler
from func.model_arch_model10_plus_tou import SegAirwayModel, AirwayClassifier
from func.enhanced_airway_model import EnhancedAirwaySegmentation
from func.loss_func import dice_loss_weights, dice_accuracy, dice_loss_power_weights

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
import os
import time
from torch.utils.tensorboard import SummaryWriter

# 使用 time 模块生成时间戳
current_time = time.strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('runs_seg', current_time)
print("log dictionary:", log_dir)
writer = SummaryWriter(log_dir=log_dir)

# 设置随机种子
seed = 1419
from func.load_dataset import seed_everything

seed_everything(seed)

# settings
# ----------
model_name = "BAS_enhanced_segmentation_with_classifier"

# Load paths for pretrained models
segmentation_model_path = '/home/ubuntu/ICME/checkpoint/BAS_model10-200epoch_-ronghe-plus-tou-xiuzheng_599.pkl'
classifier_model_path = '/home/ubuntu/ICME/runs/20250318-140331/400BAS_model10-200epoch_-ronghe-xiuzheng_classfier_5.pkl'

pkl_path = "./dataset_info/3dataset_info_BAStrain_merged_dicescores_wclusters_extended.pkl"
npy_base_dir = "/home/ubuntu/shared-data/Precrop_dataset/Precrop_dataset_for_BAS"

learning_rate = 5e-6  # Lower learning rate for fine-tuning
max_epoch = 200
num_samples_of_each_epoch = 2000
batch_size = 4  # Reduce batch size if needed due to increased model size
train_file_format = '.npy'
num_workers = 4
prefetch_factor = 16

crop_size = (128, 128, 128)
windowMin_CT_img_HU = -1000
windowMax_CT_img_HU = 600
model_save_freq = 1

# ----------

# Initialize base models
base_segmentation_model = SegAirwayModel(in_channels=1, out_channels=2)
classifier_model = AirwayClassifier(num_classes=21)

# Load pretrained weights
if os.path.exists(segmentation_model_path):
    print("Loading segmentation model from " + str(segmentation_model_path))
    checkpoint = torch.load(segmentation_model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    # 移除键名中的"module."前缀
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    base_segmentation_model.load_state_dict(new_state_dict, strict=True)
else:
    print("WARNING: Segmentation model checkpoint not found!")

if os.path.exists(classifier_model_path):
    print("Loading classifier model from " + str(classifier_model_path))
    checkpoint = torch.load(classifier_model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    # 移除键名中的"module."前缀
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    classifier_model.load_state_dict(new_state_dict, strict=True)
else:
    print("WARNING: Classifier model checkpoint not found!")

# Initialize enhanced model
model = EnhancedAirwaySegmentation(base_segmentation_model, classifier_model)


# 冻结分类器的大部分参数，只保留最后几层可训练
# 注意：访问方式会根据模型是否被DataParallel包装而不同
def freeze_classifier_params(model):
    # 处理 DataParallel 封装的情况
    if isinstance(model, torch.nn.DataParallel):
        classifier = model.module.classifier
    else:
        classifier = model.classifier

    # 冻结分类器参数
    for name, param in classifier.named_parameters():
        # 只保留最后几层可训练
        if 'fc' not in name and 'final' not in name:
            param.requires_grad = False
        else:
            print(f"Keeping trainable: {name}")


# 先不封装为DataParallel，以便直接访问子模块
freeze_classifier_params(model)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-7)

# Load dataset
adapter = PKLtoNPYAdapter(pkl_path, npy_base_dir)
data_dict = adapter.get_adapted_dict()
train_file_format = '.npy'
print("数据存储表转化完成")

cache_dir = "/home/ubuntu/shared-data/BAS/train/.airway_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

train_dataset = airway_dataset(data_dict, cache_dir=cache_dir, add_hierarchy_level=True, model_type="full")
train_dataset.set_para(file_format=train_file_format,
                       crop_size=crop_size,
                       windowMin=windowMin_CT_img_HU,
                       windowMax=windowMax_CT_img_HU,
                       need_tensor_output=True,
                       need_transform=True)

print('Total epoch: ' + str(max_epoch))

# Cross entropy loss for classification
cls_criterion = torch.nn.CrossEntropyLoss()
# Set up data loader for this epoch
sampler = ResetableRandomSampler(train_dataset,
                        num_samples=min(num_samples_of_each_epoch, len(data_dict)),
                        replacement=False)

dataset_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=num_workers,
                            pin_memory=True,
                            prefetch_factor=prefetch_factor,
                            persistent_workers=(num_workers > 1))

start_time = time.time()

# 跟踪模型是否开始稳定，避免在不稳定阶段使用分类加权
stable_training = False
cls_acc_window = []  # 存储最近的分类准确率
cls_acc_threshold = 0.7  # 分类准确率阈值，超过此值启用基于分类的权重调整
window_size = 20  # 窗口大小，用于判断模型稳定性

for ith_epoch in range(0, max_epoch):
    len_dataset_loader = len(dataset_loader)

    # 重设数据加载器
    sampler.set_epoch(ith_epoch)  # 如果需要每个epoch随机重置采样

    # Training loop
    model.train()
    epoch_loss = 0.0
    epoch_acc_seg = 0.0
    epoch_acc_seg2 = 0.0
    epoch_acc_gaussian = 0.0
    epoch_cls_acc = 0.0

    # 重置有气道样本的累积统计
    airway_samples_count = 0
    airway_acc_sum = 0.0
    airway_acc2_sum = 0.0

    # 重置分类统计
    correct_cls = 0
    total_cls = 0

    for ith_batch, batch in enumerate(dataset_loader):
        img_input = batch['image'].float().to(device)
        gaussian_input = batch['gaussian2'].float().to(device)  # Center connectivity
        gaussian_input_background = 1 - gaussian_input

        groundtruth_foreground = batch['label'].float().to(device)
        groundtruth_background = 1 - groundtruth_foreground

        # 获取分类目标
        if 'hierarchy_level' in batch:
            targets = batch['hierarchy_level'].long().to(device)
        else:
            # 创建默认目标（假设0是无气道类别）
            targets = torch.zeros(img_input.size(0), dtype=torch.long, device=device)

        # Calculate class weights for balanced loss
        fore_pix_num = torch.sum(groundtruth_foreground)
        back_pix_num = torch.sum(groundtruth_background)
        fore_pix_per = fore_pix_num / (fore_pix_num + back_pix_num)
        back_pix_per = back_pix_num / (fore_pix_num + back_pix_num)

        weights = (torch.exp(back_pix_per) / (torch.exp(fore_pix_per) + torch.exp(back_pix_per)) *
                   torch.eq(groundtruth_foreground, 1).float() +
                   torch.exp(fore_pix_per) / (torch.exp(fore_pix_per) + torch.exp(back_pix_per)) *
                   torch.eq(groundtruth_foreground, 0).float()).to(device)

        # Forward pass
        img_output, img_output2, gaussian_output, cls_logits = model(img_input)

        # 计算基础损失
        loss_seg = dice_loss_weights(img_output[:, 0, :, :, :], groundtruth_background, weights) + \
                   dice_loss_power_weights(img_output[:, 1, :, :, :], groundtruth_foreground, weights, alpha=2)

        loss_seg2 = dice_loss_weights(img_output2[:, 0, :, :, :], groundtruth_background, weights) + \
                    dice_loss_power_weights(img_output2[:, 1, :, :, :], groundtruth_foreground, weights, alpha=2)

        loss_attention = dice_loss_weights(gaussian_output[:, 0, :, :, :], gaussian_input_background, weights) + \
                         dice_loss_power_weights(gaussian_output[:, 1, :, :, :], gaussian_input, weights, alpha=2)

        # 分类损失
        if 'hierarchy_level' in batch:
            loss_cls = cls_criterion(cls_logits, targets)
        else:
            loss_cls = torch.tensor(0.0, device=device)

        # 分类预测和准确率计算
        _, predicted = cls_logits.max(1)
        batch_correct = predicted.eq(targets).sum().item()
        batch_total = targets.size(0)
        batch_acc = batch_correct / batch_total if batch_total > 0 else 0

        # 更新整体分类统计
        correct_cls += batch_correct
        total_cls += batch_total
        current_cls_acc = correct_cls / total_cls if total_cls > 0 else 0

        # 更新滑动窗口的分类准确率
        cls_acc_window.append(batch_acc)
        if len(cls_acc_window) > window_size:
            cls_acc_window.pop(0)

        # 检查是否进入稳定训练阶段
        if len(cls_acc_window) >= window_size and np.mean(cls_acc_window) > cls_acc_threshold:
            stable_training = True

        # 仅当进入稳定训练阶段时应用样本权重
        if stable_training and fore_pix_per.item() < 0.01:  # 当气道像素占比非常小时
            # 对无气道样本降低权重 (根据分类结果)
            no_airway_indices = (predicted == 0)
            sample_weight = 0.3  # 无气道样本权重

            # 创建批次样本权重向量
            batch_weights = torch.ones(img_input.size(0), device=device)
            batch_weights[no_airway_indices] = sample_weight

            # 调整损失
            loss_seg = loss_seg * batch_weights.mean()
            loss_seg2 = loss_seg2 * batch_weights.mean()
            loss_attention = loss_attention * batch_weights.mean()

        # 动态调整分类损失权重
        cls_weight = 0.1
        if stable_training:
            # 当分类准确率高时，降低分类损失权重
            cls_weight = max(0.01, 0.1 * (1.0 - current_cls_acc))

        # 组合损失
        loss = loss_seg + loss_seg2 + loss_attention + cls_weight * loss_cls

        # 计算分割准确率
        accuracy = dice_accuracy(img_output[:, 1, :, :, :], groundtruth_foreground)
        accuracy2 = dice_accuracy(img_output2[:, 1, :, :, :], groundtruth_foreground)
        accuracy_gaussian = dice_accuracy(gaussian_output[:, 1, :, :, :], gaussian_input)

        # 更高效地计算有气道样本的Dice分数 - 仅在有需要和有足够气道像素时计算
        airway_acc = 0.0
        airway_acc2 = 0.0
        if stable_training and fore_pix_per.item() > 0.005:  # 只在有足够气道像素时计算
            # 获取有气道像素的样本
            airway_samples = (fore_pix_per > 0.005)
            if airway_samples:
                airway_samples_count += 1
                airway_acc = accuracy.item()
                airway_acc_sum += airway_acc
                airway_acc2 = accuracy2.item()
                airway_acc2_sum += airway_acc2

        # 计算有气道样本的平均准确率
        avg_airway_acc = airway_acc_sum / max(1, airway_samples_count)
        avg_airway_acc2 = airway_acc2_sum / max(1, airway_samples_count)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics for epoch averaging
        epoch_loss += loss.item()
        epoch_acc_seg += accuracy.item()
        epoch_acc_seg2 += accuracy2.item()
        epoch_acc_gaussian += accuracy_gaussian.item()
        epoch_cls_acc += batch_acc

        # 每50个批次打印一次详细信息
        if ith_batch % 50 == 0 or ith_batch == len_dataset_loader - 1:
            # Log progress
            time_consumption = time.time() - start_time

            print(
                "epoch [{0}/{1}]\t"
                "batch [{2}/{3}]\t"
                "time(sec) {time:.2f}\t"
                "loss {loss:.4f}\t"
                "loss_seg {loss_seg:.4f}\t"
                "acc_seg {acc_seg:.2f}%\t"
                "acc_airway {acc_air:.2f}%\t"
                "loss_seg2 {loss_seg2:.4f}\t"
                "acc_seg2 {acc_seg2:.2f}%\t"
                "loss_att {loss_att:.4f}\t"
                "acc_att {acc_att:.2f}%\t"
                "cls_acc {cls_acc:.2f}%\t"
                "stable {stable}\t"
                "fore_pix {fore_pix:.2f}%\t".format(
                    ith_epoch + 1,
                    max_epoch,
                    ith_batch,
                    len_dataset_loader,
                    time=time_consumption,
                    loss=loss.item(),
                    loss_seg=loss_seg.item(),
                    acc_seg=accuracy.item() * 100,
                    acc_air=avg_airway_acc * 100,
                    loss_seg2=loss_seg2.item(),
                    acc_seg2=accuracy2.item() * 100,
                    loss_att=loss_attention.item(),
                    acc_att=accuracy_gaussian.item() * 100,
                    cls_acc=current_cls_acc * 100,
                    stable=stable_training,
                    fore_pix=fore_pix_per.item() * 100))

        # TensorBoard logging - 减少记录频率以提高性能
        if ith_batch % 10 == 0:  # 每10个批次记录一次
            global_step = ith_epoch * len_dataset_loader + ith_batch
            writer.add_scalar('Loss/total', loss.item(), global_step=global_step)
            writer.add_scalar('Loss/segmentation1', loss_seg.item(), global_step=global_step)
            writer.add_scalar('Loss/segmentation2', loss_seg2.item(), global_step=global_step)
            writer.add_scalar('Loss/gaussian', loss_attention.item(), global_step=global_step)
            writer.add_scalar('Loss/classification', loss_cls.item(), global_step=global_step)

            writer.add_scalar('Accuracy/segmentation1', accuracy.item() * 100, global_step=global_step)
            writer.add_scalar('Accuracy/segmentation2', accuracy2.item() * 100, global_step=global_step)
            writer.add_scalar('Accuracy/gaussian', accuracy_gaussian.item() * 100, global_step=global_step)
            writer.add_scalar('Accuracy/classification', current_cls_acc * 100, global_step=global_step)

            if airway_samples_count > 0:
                writer.add_scalar('Accuracy/airway_seg', avg_airway_acc * 100, global_step=global_step)
                writer.add_scalar('Accuracy/airway_seg2', avg_airway_acc2 * 100, global_step=global_step)

            writer.add_scalar('Training/foreground_pixel_ratio', fore_pix_per.item() * 100, global_step=global_step)
            writer.add_scalar('Training/stable_training', int(stable_training), global_step=global_step)
            writer.add_scalar('Training/cls_weight', cls_weight, global_step=global_step)

    # End of epoch - update learning rate
    scheduler.step()

    # Log epoch averages
    epoch_loss /= len_dataset_loader
    epoch_acc_seg /= len_dataset_loader
    epoch_acc_seg2 /= len_dataset_loader
    epoch_acc_gaussian /= len_dataset_loader
    epoch_cls_acc /= len_dataset_loader

    # 计算有气道样本的平均准确率
    epoch_airway_acc = airway_acc_sum / max(1, airway_samples_count)
    epoch_airway_acc2 = airway_acc2_sum / max(1, airway_samples_count)

    writer.add_scalar('Epoch/loss', epoch_loss, global_step=ith_epoch)
    writer.add_scalar('Epoch/accuracy_seg', epoch_acc_seg * 100, global_step=ith_epoch)
    writer.add_scalar('Epoch/accuracy_seg2', epoch_acc_seg2 * 100, global_step=ith_epoch)
    writer.add_scalar('Epoch/accuracy_gaussian', epoch_acc_gaussian * 100, global_step=ith_epoch)
    writer.add_scalar('Epoch/accuracy_cls', epoch_cls_acc * 100, global_step=ith_epoch)

    if airway_samples_count > 0:
        writer.add_scalar('Epoch/accuracy_airway', epoch_airway_acc * 100, global_step=ith_epoch)
        writer.add_scalar('Epoch/accuracy_airway2', epoch_airway_acc2 * 100, global_step=ith_epoch)

    writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=ith_epoch)
    writer.add_scalar('Epoch/stable_training', int(stable_training), global_step=ith_epoch)

    # 打印epoch摘要
    print(f"=== Epoch {ith_epoch + 1}/{max_epoch} Summary ===")
    print(f"Loss: {epoch_loss:.4f}")
    print(f"Seg Accuracy: {epoch_acc_seg * 100:.2f}%")
    print(f"Seg2 Accuracy: {epoch_acc_seg2 * 100:.2f}%")
    print(f"Gaussian Accuracy: {epoch_acc_gaussian * 100:.2f}%")
    print(f"Classification Accuracy: {epoch_cls_acc * 100:.2f}%")
    if airway_samples_count > 0:
        print(f"Airway Seg Accuracy: {epoch_airway_acc * 100:.2f}%")
        print(f"Airway Seg2 Accuracy: {epoch_airway_acc2 * 100:.2f}%")
    print(f"Stable Training: {stable_training}")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]}")

    # Save model
    if (ith_epoch + 1) % model_save_freq == 0:
        print(f'Epoch: {ith_epoch} - Saving model')
        model.to(torch.device('cpu'))
        model_save_path = os.path.join(writer.log_dir, f"{model_name}_{ith_epoch}.pkl")
        torch.save({'model_state_dict': model.state_dict()}, model_save_path)
        model.to(device)

writer.close()
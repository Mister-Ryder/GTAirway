# train
from func.load_dataset import airway_dataset, PKLtoNPYAdapter
from func.model_arch_model10_plus_tou import SegAirwayModel, AirwayClassifier
from func.enhanced_airway_model import EnhancedAirwaySegmentation
from func.loss_func import dice_loss_weights, dice_accuracy, dice_loss_power_weights

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
import os
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

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

# 冻结分类器的主要部分参数，只保留最后几层可训练
# 注意：如果使用DataParallel，需要在包装前冻结参数
for name, param in model.classifier.named_parameters():
    if 'fc' not in name and 'final' not in name:  # 根据实际层名调整
        param.requires_grad = False

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 将模型参数分为两组，分别设置不同学习率
segmentation_params = []
classification_params = []

for name, param in model.named_parameters():
    if 'classifier' in name and param.requires_grad:
        classification_params.append(param)
    elif param.requires_grad:
        segmentation_params.append(param)

# 使用不同学习率的优化器
optimizer = torch.optim.AdamW([
    {'params': segmentation_params, 'lr': learning_rate},
    {'params': classification_params, 'lr': learning_rate * 0.1}  # 分类器使用较小学习率
], weight_decay=1e-4)

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
sampler = RandomSampler(train_dataset,
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
# 监控分类准确率的历史
cls_acc_history = []
cls_acc_threshold = 0.85  # 分类准确率阈值，超过此值才启用基于分类的权重调整

for ith_epoch in range(0, max_epoch):
    len_dataset_loader = len(dataset_loader)

    # Training loop
    model.train()
    epoch_loss = 0.0
    epoch_acc_seg = 0.0
    epoch_acc_seg2 = 0.0
    epoch_acc_gaussian = 0.0
    epoch_acc_cls = 0.0
    epoch_acc_seg_airway = 0.0
    epoch_acc_seg2_airway = 0.0

    # 重置分类统计
    correct_cls = 0
    total_cls = 0

    # 用于统计有气道样本的准确率
    airway_dice_sum = 0.0
    airway_dice_count = 0
    airway_dice2_sum = 0.0
    airway_dice2_count = 0

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

        # 计算分类准确率
        _, predicted = cls_logits.max(1)
        correct_batch = predicted.eq(targets).sum().item()
        correct_cls += correct_batch
        total_cls += targets.size(0)
        current_cls_accuracy = correct_cls / total_cls if total_cls > 0 else 0

        # 使用分类结果创建样本权重
        with torch.no_grad():
            # 默认所有样本权重为1
            sample_weights = torch.ones(img_input.size(0), device=device)

            # 基于分类准确率和历史决定是否应用基于分类的样本权重
            use_cls_weights = len(cls_acc_history) >= 5 and np.mean(cls_acc_history[-5:]) >= cls_acc_threshold

            if use_cls_weights:
                # 0类通常是无气道类别
                no_airway_indices = (predicted == 0)
                sample_weights[no_airway_indices] = 0.2  # 无气道样本权重降低

            # 转换为适合损失计算的形状
            sample_weights = sample_weights.view(-1, 1, 1, 1, 1)

        # 计算带样本权重的损失
        loss_seg = (dice_loss_weights(img_output[:, 0, :, :, :], groundtruth_background, weights) +
                    dice_loss_power_weights(img_output[:, 1, :, :, :], groundtruth_foreground, weights,
                                            alpha=2)) * sample_weights

        loss_seg2 = (dice_loss_weights(img_output2[:, 0, :, :, :], groundtruth_background, weights) +
                     dice_loss_power_weights(img_output2[:, 1, :, :, :], groundtruth_foreground, weights,
                                             alpha=2)) * sample_weights

        loss_attention = (dice_loss_weights(gaussian_output[:, 0, :, :, :], gaussian_input_background, weights) +
                          dice_loss_power_weights(gaussian_output[:, 1, :, :, :], gaussian_input, weights,
                                                  alpha=2)) * sample_weights

        # 取平均
        loss_seg = loss_seg.mean()
        loss_seg2 = loss_seg2.mean()
        loss_attention = loss_attention.mean()

        # 分类损失
        loss_cls = cls_criterion(cls_logits, targets)

        # 动态调整分类损失权重
        cls_weight = 0.1
        if total_cls > 0:
            cls_weight = min(0.1, 0.05 * (1.0 - current_cls_accuracy))

        # 组合损失
        loss = loss_seg + loss_seg2 + loss_attention + cls_weight * loss_cls

        # 计算所有样本的准确率
        accuracy = dice_accuracy(img_output[:, 1, :, :, :], groundtruth_foreground)
        accuracy2 = dice_accuracy(img_output2[:, 1, :, :, :], groundtruth_foreground)
        accuracy_gaussian = dice_accuracy(gaussian_output[:, 1, :, :, :], gaussian_input)

        # 计算仅有气道样本的准确率
        with torch.no_grad():
            airway_samples = (predicted != 0)
            if torch.sum(airway_samples) > 0:
                airway_acc = dice_accuracy(
                    img_output[airway_samples, 1, :, :, :],
                    groundtruth_foreground[airway_samples]
                ).item()
                airway_dice_sum += airway_acc
                airway_dice_count += 1

                airway_acc2 = dice_accuracy(
                    img_output2[airway_samples, 1, :, :, :],
                    groundtruth_foreground[airway_samples]
                ).item()
                airway_dice2_sum += airway_acc2
                airway_dice2_count += 1

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics for epoch averaging
        epoch_loss += loss.item()
        epoch_acc_seg += accuracy.item()
        epoch_acc_seg2 += accuracy2.item()
        epoch_acc_gaussian += accuracy_gaussian.item()
        epoch_acc_cls += current_cls_accuracy

        # 计算有气道样本的平均准确率
        airway_acc_avg = airway_dice_sum / airway_dice_count if airway_dice_count > 0 else 0
        airway_acc2_avg = airway_dice2_sum / airway_dice2_count if airway_dice2_count > 0 else 0

        # Log progress
        time_consumption = time.time() - start_time

        print(
            "epoch [{0}/{1}]\t"
            "batch [{2}/{3}]\t"
            "time(sec) {time:.2f}\t"
            "loss {loss:.4f}\t"
            "loss_seg {loss_seg:.4f}\t"
            "acc_seg {acc_seg:.2f}%\t"
            "acc_seg_air {acc_seg_air:.2f}%\t"
            "loss_seg2 {loss_seg2:.4f}\t"
            "acc_seg2 {acc_seg2:.2f}%\t"
            "acc_seg2_air {acc_seg2_air:.2f}%\t"
            "loss_att {loss_att:.4f}\t"
            "acc_att {acc_att:.2f}%\t"
            "cls_acc {cls_acc:.2f}%\t"
            "fore pix {fore_pix_percentage:.2f}%\t".format(
                ith_epoch + 1,
                max_epoch,
                ith_batch,
                len_dataset_loader,
                time=time_consumption,
                loss=loss.item(),
                loss_seg=loss_seg.item(),
                acc_seg=accuracy.item() * 100,
                acc_seg_air=airway_acc_avg * 100,
                loss_seg2=loss_seg2.item(),
                acc_seg2=accuracy2.item() * 100,
                acc_seg2_air=airway_acc2_avg * 100,
                loss_att=loss_attention.item(),
                acc_att=accuracy_gaussian.item() * 100,
                cls_acc=current_cls_accuracy * 100,
                fore_pix_percentage=fore_pix_per * 100))

        # TensorBoard logging
        global_step = ith_epoch * len_dataset_loader + ith_batch
        writer.add_scalar('Loss/total', loss.item(), global_step=global_step)
        writer.add_scalar('Loss/segmentation1', loss_seg.item(), global_step=global_step)
        writer.add_scalar('Loss/segmentation2', loss_seg2.item(), global_step=global_step)
        writer.add_scalar('Loss/gaussian', loss_attention.item(), global_step=global_step)
        writer.add_scalar('Loss/classification', loss_cls.item(), global_step=global_step)
        writer.add_scalar('Loss/cls_weight', cls_weight, global_step=global_step)

        writer.add_scalar('Accuracy/segmentation1', accuracy.item() * 100, global_step=global_step)
        writer.add_scalar('Accuracy/segmentation2', accuracy2.item() * 100, global_step=global_step)
        writer.add_scalar('Accuracy/gaussian', accuracy_gaussian.item() * 100, global_step=global_step)
        writer.add_scalar('Accuracy/classification', current_cls_accuracy * 100, global_step=global_step)

        if airway_dice_count > 0:
            writer.add_scalar('Accuracy/segmentation1_airway_only', airway_acc_avg * 100, global_step=global_step)
        if airway_dice2_count > 0:
            writer.add_scalar('Accuracy/segmentation2_airway_only', airway_acc2_avg * 100, global_step=global_step)

        writer.add_scalar('Training/sample_weight_applied', float(use_cls_weights), global_step=global_step)
        writer.add_scalar('Training/airway_pixel_percentage', fore_pix_per * 100, global_step=global_step)

    # 记录分类准确率历史
    epoch_acc_cls /= len_dataset_loader
    cls_acc_history.append(epoch_acc_cls)

    # End of epoch - update learning rate
    scheduler.step()

    # Log epoch averages
    epoch_loss /= len_dataset_loader
    epoch_acc_seg /= len_dataset_loader
    epoch_acc_seg2 /= len_dataset_loader
    epoch_acc_gaussian /= len_dataset_loader
    epoch_acc_seg_airway = airway_dice_sum / airway_dice_count if airway_dice_count > 0 else 0
    epoch_acc_seg2_airway = airway_dice2_sum / airway_dice2_count if airway_dice2_count > 0 else 0

    writer.add_scalar('Epoch/loss', epoch_loss, global_step=ith_epoch)
    writer.add_scalar('Epoch/accuracy_seg', epoch_acc_seg * 100, global_step=ith_epoch)
    writer.add_scalar('Epoch/accuracy_seg2', epoch_acc_seg2 * 100, global_step=ith_epoch)
    writer.add_scalar('Epoch/accuracy_gaussian', epoch_acc_gaussian * 100, global_step=ith_epoch)
    writer.add_scalar('Epoch/accuracy_classification', epoch_acc_cls * 100, global_step=ith_epoch)
    writer.add_scalar('Epoch/accuracy_seg_airway', epoch_acc_seg_airway * 100, global_step=ith_epoch)
    writer.add_scalar('Epoch/accuracy_seg2_airway', epoch_acc_seg2_airway * 100, global_step=ith_epoch)
    writer.add_scalar('Epoch/learning_rate', scheduler.get_last_lr()[0], global_step=ith_epoch)

    # 记录是否使用了分类权重
    use_cls_weights = len(cls_acc_history) >= 5 and np.mean(cls_acc_history[-5:]) >= cls_acc_threshold
    writer.add_scalar('Epoch/using_cls_weights', float(use_cls_weights), global_step=ith_epoch)

    # Clean up
    del dataset_loader

    # Save model
    if (ith_epoch + 1) % model_save_freq == 0:
        print('Epoch: ' + str(ith_epoch) + ' - Saving model')
        model.to(torch.device('cpu'))
        model_save_path = os.path.join(writer.log_dir, f"{model_name}_{ith_epoch}.pkl")
        torch.save({'model_state_dict': model.state_dict()}, model_save_path)
        model.to(device)

writer.close()
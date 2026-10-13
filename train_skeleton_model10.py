# train
from func.load_dataset import airway_dataset
# from func.unet_3d_basic import UNet3D_basic
from func.model_arch_model10 import SegAirwayModel
from func.loss_func import dice_loss_weights, dice_accuracy, dice_loss_power_weights, dice_loss, dice_loss_power, \
    mae_loss, mse_loss
from func.ulti import load_obj

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import random
import os
import time
import gc
from evaluate_skeleton import evaluation
from torch.utils.tensorboard import SummaryWriter

# 使用 time 模块生成时间戳
current_time = time.strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join('runs', current_time)
writer = SummaryWriter(log_dir=log_dir)


# 设置随机种子
seed = 114514
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用了多GPU
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(seed)


class DualDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        if len(dataset1) != len(dataset2):
            raise ValueError("Length of dataset1 and dataset2 should be the same.")

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        idx2 = idx
        if isinstance(idx, slice):
            idx2 = random.randrange(*idx.indices(len(self.dataset2)))
        return self.dataset1[idx], self.dataset2[idx2]


# settings
# ----------
model_name = "BAS_plus_model10-200epoch_jiaoti"
save_path = '/home/ubuntu/GTAirway-main-BAS/checkpoint/' + model_name + '.pkl'
print(save_path)
need_resume = False #注意修改模型保存命名和测试周期设置
load_path =  ''  #""#

path_dataset_info_airways_skeleton_of_generation_plus = \
    "/home/ubuntu/GTAirway-main-BAS/dataset_info/train_dataset_info_BAS_Skeleton_crops_128_extended_generation_plus"
# path_dataset_info_more_focus_on_airways_skeleton_of_low_generation = \
#     "/home/ubuntu/GTAirway-main-BAS/dataset_info/train_dataset_info_BAS_Skeleton_crops_128_extended_more_low_generation_10"
# path_dataset_info_more_focus_on_airways_skeleton_of_high_generation = \
#     "/home/ubuntu/GTAirway-main-BAS/dataset_info/train_dataset_info_BAS_Skeleton_crops_128_extended_more_high_generation_1"

learning_rate = 1e-5
max_epoch = 200  # 50
freq_switch_of_train_mode_high_low_generation = 2
num_samples_of_each_epoch = 2000 #plus的总数为4395+
batch_size = 16
train_file_format = '.npy'
crop_size = (128, 128, 128)  # (32, 128, 128)
windowMin_CT_img_HU = -1000  # min of CT image HU value
windowMax_CT_img_HU = 600  # max of CT image HU value
model_save_freq = 1
num_workers = 4

# ----------

# init model
model = SegAirwayModel(in_channels=1, out_channels=2)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 包装你的模型以使用多个GPU
    model = torch.nn.DataParallel(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# if need_resume and os.path.exists(load_path):
#     print("resume model from " + str(load_path))
#     print(1)
#     checkpoint = torch.load(load_path)
#     # 获取完整的模型状态字典
#     full_state_dict = checkpoint['model_state_dict']
#     # 创建一个新的状态字典，只包含 encoders 和 decoders 层的参数
#     filtered_state_dict = {k: v for k, v in full_state_dict.items() if 'encoders' in k or 'decoders' in k}
#     # 加载过滤后的状态字典到模型中
#     # strict=False 允许模型中的其他层保持原始初始化状态，不会因为状态字典中缺少这些层的参数而报错
#     model.load_state_dict(filtered_state_dict, strict=False)
#     # model.load_state_dict(checkpoint['model_state_dict'], strict=False)

if need_resume and os.path.exists(load_path):
    print("resume model from " + str(load_path))
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# dataset
dataset_info_airways_skeleton_of_generation_plus = load_obj(
    path_dataset_info_airways_skeleton_of_generation_plus)
train_dataset_airways_skeleton_of_generation_plus = airway_dataset(
    dataset_info_airways_skeleton_of_generation_plus)
train_dataset_airways_skeleton_of_generation_plus.set_para(file_format=train_file_format,
                                                           crop_size=crop_size, \
                                                           windowMin=windowMin_CT_img_HU,
                                                           windowMax=windowMax_CT_img_HU,
                                                           need_tensor_output=True, need_transform=True)

# dataset_info_more_focus_on_airways_skeleton_of_low_generation = load_obj(
#     path_dataset_info_more_focus_on_airways_skeleton_of_low_generation)
# train_dataset_more_focus_on_airways_skeleton_of_low_generation = airway_dataset(
#     dataset_info_more_focus_on_airways_skeleton_of_low_generation)
# train_dataset_more_focus_on_airways_skeleton_of_low_generation.set_para(file_format=train_file_format,
#                                                                         crop_size=crop_size, \
#                                                                         windowMin=windowMin_CT_img_HU,
#                                                                         windowMax=windowMax_CT_img_HU,
#                                                                         need_tensor_output=True, need_transform=True)
# 
# dataset_info_more_focus_on_airways_skeleton_of_high_generation = load_obj(
#     path_dataset_info_more_focus_on_airways_skeleton_of_high_generation)
# train_dataset_more_focus_on_airways_skeleton_of_high_generation = airway_dataset(
#     dataset_info_more_focus_on_airways_skeleton_of_high_generation)
# train_dataset_more_focus_on_airways_skeleton_of_high_generation.set_para(file_format=train_file_format,
#                                                                          crop_size=crop_size, \
#                                                                          windowMin=windowMin_CT_img_HU,
#                                                                          windowMax=windowMax_CT_img_HU,
#                                                                          need_tensor_output=True, need_transform=True)

print('total epoch: ' + str(max_epoch))

start_time = time.time()
change = [1, 1]  # 应先进行自发联合训练，再利用标签集中优化label的分割。
for ith_epoch in range(0, max_epoch):

    print("train_airways_plus")
    sampler_of_airways_of_low_generation = RandomSampler(
        train_dataset_airways_skeleton_of_generation_plus, \
        num_samples=min(num_samples_of_each_epoch,
                        len(dataset_info_airways_skeleton_of_generation_plus)),
        replacement=True)
    dataset_loader = DataLoader(train_dataset_airways_skeleton_of_generation_plus, \
                                batch_size=batch_size, sampler=sampler_of_airways_of_low_generation,
                                num_workers=num_workers, \
                                pin_memory=True, persistent_workers=(num_workers > 1))

    # if np.floor(ith_epoch / freq_switch_of_train_mode_high_low_generation) % 2 == 0 or \
    #         ith_epoch >= (max_epoch - freq_switch_of_train_mode_high_low_generation):
    #     print("train_more_focus_on_airways_of_low_density")
    #     sampler_of_airways_of_low_generation = RandomSampler(
    #         train_dataset_more_focus_on_airways_skeleton_of_low_generation, \
    #         num_samples=min(num_samples_of_each_epoch,
    #                         len(dataset_info_airways_skeleton_of_generation_plus)),
    #         replacement=True)
    #     dataset_loader = DataLoader(train_dataset_more_focus_on_airways_skeleton_of_low_generation, \
    #                                 batch_size=batch_size, sampler=sampler_of_airways_of_low_generation,
    #                                 num_workers=num_workers, \
    #                                 pin_memory=True, persistent_workers=(num_workers > 1))
    #
    # else:
    #     print("train_more_focus_on_airways_of_high_density")
    #     sampler_of_airways_of_high_generation = RandomSampler(
    #         train_dataset_more_focus_on_airways_skeleton_of_high_generation, \
    #         num_samples=min(num_samples_of_each_epoch,
    #                         len(dataset_info_more_focus_on_airways_skeleton_of_high_generation)), replacement=True)
    #     dataset_loader = DataLoader(train_dataset_more_focus_on_airways_skeleton_of_high_generation, \
    #                                 batch_size=batch_size, sampler=sampler_of_airways_of_high_generation,
    #                                 num_workers=num_workers, \
    #                                 pin_memory=True, persistent_workers=(num_workers > 1))

    len_dataset_loader = len(dataset_loader)

    for ith_batch, batch in enumerate(dataset_loader):
        # concatenated_input = torch.cat((batch['image'], batch['gaussian']), dim=1)
        # concatenated_input = concatenated_input.float().to(device)
        img_input = batch['image'].float().to(device)
        # skeleton_input = batch['skeleton'].float().to(device)

        p = np.random.random()
        if p > 0.75:
            gaussian_input = batch['gaussian'].float().to(device)  # bianyuanjingque
        else:
            gaussian_input = batch['gaussian2'].float().to(device)  # zhongxinliantong

        # distance_input = batch['distance'].float().to(device)
        gaussian_input_background = 1 - gaussian_input
        # # 反向
        # gaussian_input_background = torch.where(gaussian_input_background == 24,
        #                                         torch.zeros_like(gaussian_input_background), gaussian_input_background)
        # # torch.where(condition, x, y) → Tensor,x if condition else y,因此上面的代码是将gaussian_input_background中的1替换为0
        # gaussian_input_background_background = 24 - gaussian_input_background
        #
        groundtruth_foreground = batch['label'].float().to(device)
        groundtruth_background = 1 - groundtruth_foreground

        fore_pix_num = torch.sum(groundtruth_foreground)
        back_pix_num = torch.sum(groundtruth_background)
        fore_pix_per = fore_pix_num / (fore_pix_num + back_pix_num)
        back_pix_per = back_pix_num / (fore_pix_num + back_pix_num)

        # if fore_pix_per * 100 <= 0.005:
        #     # print(0)
        #     continue

        weights = (torch.exp(back_pix_per) / (torch.exp(fore_pix_per) + torch.exp(back_pix_per)) * torch.eq(
            groundtruth_foreground, 1).float() + \
                   torch.exp(fore_pix_per) / (torch.exp(fore_pix_per) + torch.exp(back_pix_per)) * torch.eq(
                    groundtruth_foreground, 0).float()).to(device)

        # model训练
        img_output, img_output2, gaussian_output = model(img_input)
        # img_output, skl_output= model(concatenated_input)

        # loss_skl=mse_loss(skl_output[:, 1, :, :, :],skeleton_input)

        loss_seg = dice_loss_weights(img_output[:, 0, :, :, :], groundtruth_background, weights) + \
                   dice_loss_power_weights(img_output[:, 1, :, :, :], groundtruth_foreground, weights, alpha=2)
        accuracy = dice_accuracy(img_output[:, 1, :, :, :], groundtruth_foreground)
        loss_seg2 = dice_loss_weights(img_output2[:, 0, :, :, :], groundtruth_background, weights) + \
                    dice_loss_power_weights(img_output2[:, 1, :, :, :], groundtruth_foreground, weights, alpha=2)
        accuracy2 = dice_accuracy(img_output2[:, 1, :, :, :], groundtruth_foreground)

        # loss_attention = dice_loss_weights(gaussian_output[:, 0, :, :, :], gaussian_input_background, weights) + \
        #            dice_loss_power_weights(gaussian_output[:, 1, :, :, :], gaussian_input, weights, alpha=2)
        # accuracy_without_gaussian = dice_accuracy(gaussian_output[:, 1, :, :, :], gaussian_input)
        loss_attention = dice_loss_weights(gaussian_output[:, 0, :, :, :], gaussian_input_background,
                                           weights) + \
                         dice_loss_power_weights(gaussian_output[:, 1, :, :, :], gaussian_input, weights,
                                                 alpha=2)
        accuracy_without_gaussian = dice_accuracy(gaussian_output[:, 1, :, :, :], gaussian_input)

        # loss_distance= ssim_loss(img_output[:, 3, :, :, :],distance_input)
        #
        # if accuracy_without_gaussian.item()>=0.9:
        #     print(accuracy_without_gaussian.item())
        #     change[0]=0
        #     print("Gaussian Train!")
        # if loss_distance.item()<=0.1:
        #     change[1]=0
        #     print("Distance Train!")

        # skl = batch['skeleton'].float().to(device)
        # gau = batch['gaussian'].float().to(device)

        # accuracy_skl = dice_accuracy(skl_output, skl)
        # accuracy_gau = dice_accuracy(gau_output, gau)

        # loss_skl = mae_loss(skl_output, skl)
        # loss_gau = mse_loss(gau_output, gau)
        # print("dice loss: ", loss.item())
        # loss = loss + loss_skl
        # print("mae loss: ", loss_skl.item())
        loss = loss_seg + loss_attention + loss_seg2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if fore_pix_per * 100 <= 0.005:
        #     print(0)
        #     optimizer.zero_grad()
        #     continue
        # else:
        #     optimizer.step()

        time_consumption = time.time() - start_time

        print(
            "epoch [{0}/{1}]\t"
            "batch [{2}/{3}]\t"
            "time(sec) {time:.2f}\t"
            "loss {loss:.4f}\t"
            "loss_seg {loss_seg:.4f}\t"
            "acc_seg {acc_seg:.2f}%\t"
            "loss_seg2 {loss_seg2:.4f}\t"
            "acc_seg2 {acc_seg2:.2f}%\t"
            "loss_att {loss_att:.4f}\t"
            "acc_att {acc_att:.2f}%\t"
            "fore pix {fore_pix_percentage:.2f}%\t"
            "back pix {back_pix_percentage:.2f}%\t".format(
                ith_epoch + 1,
                max_epoch,
                ith_batch,
                len_dataset_loader,
                time=time_consumption,
                loss=loss.item(),
                loss_seg=loss_seg.item(),
                acc_seg=accuracy.item() * 100,
                loss_seg2=loss_seg2.item(),
                acc_seg2=accuracy2.item() * 100,
                loss_att=loss_attention.item(),
                acc_att=accuracy_without_gaussian.item() * 100,
                # loss3=loss_skl.item(),"loss3 {loss3:.4f}\t"
                # acc_skl=accuracy_skl.item() * 100,
                fore_pix_percentage=fore_pix_per * 100,
                back_pix_percentage=back_pix_per * 100))
        writer.add_scalar('Loss/loss', loss.item(), global_step=ith_epoch * len_dataset_loader + ith_batch)
        writer.add_scalar('Loss/loss_seg', loss_seg.item(), global_step=ith_epoch * len_dataset_loader + ith_batch)
        writer.add_scalar('Accuracy/accuracy_seg', accuracy.item() * 100,
                          global_step=ith_epoch * len_dataset_loader + ith_batch)
        writer.add_scalar('Loss/loss_seg_2', loss_seg2.item(), global_step=ith_epoch * len_dataset_loader + ith_batch)
        writer.add_scalar('Accuracy/accuracy_seg_2', accuracy2.item() * 100,
                          global_step=ith_epoch * len_dataset_loader + ith_batch)
        writer.add_scalar('Loss/loss_att', loss_attention.item(),
                          global_step=ith_epoch * len_dataset_loader + ith_batch)
        writer.add_scalar('Accuracy/accuracy_att', accuracy_without_gaussian.item() * 100,
                          global_step=ith_epoch * len_dataset_loader + ith_batch)

    del dataset_loader
    # gc.collect()
    # gc.collect() #清理内存

    # if ith_epoch >= 50 & (ith_epoch + 1) % model_save_freq == 0:
    # 当前epoch大于100或者小于50且每隔model_save_freq个epoch做一次evaluation
    # if ith_epoch >= 1:
    #     if (ith_epoch + 1) % (5 * model_save_freq) == 0:
    #         data_path = "./test_data/image/"
    #         # save_path = "./save_path/"
    #         parsing_path = "./test_data/parsing_path/"
    #         sen1_mean, branch_mean, pre_mean = evaluation(data_path, save_path, parsing_path, model)
    #         writer.add_scalar('Evaluate/SEN', sen1_mean, global_step=ith_epoch)
    #         writer.add_scalar('Evaluate/BRANCH', branch_mean, global_step=ith_epoch)
    #         writer.add_scalar('Evaluate/PRE', pre_mean, global_step=ith_epoch)

    # gc.collect() #

    if (ith_epoch + 1) % model_save_freq == 0:
        print('epoch: ' + str(ith_epoch) + ' save model')
        model.to(torch.device('cpu'))
        # Create a descriptive save path for each model state
        model_save_path = os.path.join(writer.log_dir, model_name+'_' + str(
            ith_epoch) + '.pkl')  # save_path.replace('.pkl', '_epoch_' + str(ith_epoch) + '.pkl')
        torch.save({'model_state_dict': model.state_dict()}, model_save_path)
        # torch.save(model.state_dict(), os.path.join(writer.log_dir, f'model_epoch_{epoch}.pth'))
        model.to(device)

writer.close()
# train
from func.load_dataset import airway_dataset,ResetableRandomSampler,PKLtoNPYAdapter
# from func.unet_3d_basic import UNet3D_basic
from func.model_arch_model10_plus_tou import SegAirwayModel,AirwayClassifier
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
from torch.cuda.amp import autocast, GradScaler

# Training function
def train_classifier(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['image'].float().to(device)
        # Convert hierarchy_level to class index
        targets = batch['hierarchy_level'].long().to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return running_loss / (batch_idx + 1), accuracy

if __name__=="__main__":

    # 使用 time 模块生成时间戳
    current_time = time.strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', current_time)
    writer = SummaryWriter(log_dir=log_dir)

    # 设置随机种子 加载预训练模型要更换随机因子才能采样新的数据
    # seed = 114514
    # seed = 91314
    seed = 1419
    from func.load_dataset import seed_everything
    seed_everything(seed)


    # settings
    # ----------
    model_name = "BAS_model10-200epoch_-ronghe-xiuzheng_classfier"
    # save_path = '/home/ubuntu/ICME/checkpoint/' + model_name + '.pkl'
    # print(save_path)
    # need_resume = True #注意修改模型保存命名和测试周期设置
    # load_path =  '/home/ubuntu/GTAirway-main-BAS/runs/20240906-024934/BAS_model10-200epoch_-ronghe-xiuzheng_399.pkl'  #""#

    # path_dataset_info_airways_skeleton_of_generation_plus = \
    #     "/home/ubuntu/GTAirway-main-BAS/dataset_info/train_dataset_info_BAS_Skeleton_crops_128_extended_generation_plus2"

    learning_rate = 1e-5
    max_epoch = 100  # 50
    freq_switch_of_train_mode_high_low_generation = 2
    num_samples_of_each_epoch = 10000 #plus的总数为4395+
    batch_size = 32
    train_file_format = '.nii.gz'
    num_workers = 4
    prefetch_factor = 16

    crop_size = (128, 128, 128)  # (32, 128, 128)
    windowMin_CT_img_HU = -1000  # min of CT image HU value
    windowMax_CT_img_HU = 600  # max of CT image HU value
    model_save_freq = 1


    # ----------

    # init model
    model = AirwayClassifier()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 包装你的模型以使用多个GPU
        model = torch.nn.DataParallel(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # if need_resume and os.path.exists(load_path):
    #     print("resume model from " + str(load_path))
    #     checkpoint = torch.load(load_path)
    #     model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # dataset
    print('total epoch: ' + str(max_epoch))

    start_time = time.time()
    pkl_path = "./dataset_info/3dataset_info_BAStrain_merged_dicescores_wclusters_extended.pkl"
    npy_base_dir = "/home/ubuntu/shared-data/Precrop_dataset/Precrop_dataset_for_BAS"
    adapter = PKLtoNPYAdapter(pkl_path, npy_base_dir)
    data_dict = adapter.get_adapted_dict()
    train_file_format='.npy'
    print("转化完成")
    # data_dict = load_obj("./dataset_info/3dataset_info_BAStrain_merged_dicescores_wclusters_extended")

    cache_dir = "/home/ubuntu/shared-data/BAS/train/.airway_cache"## Remove the cache directory  rm -rf ~/.airway_cache
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    dataset = airway_dataset(data_dict, cache_dir=cache_dir,add_hierarchy_level=True)
    dataset.set_para(file_format=train_file_format,
                    crop_size=crop_size,
                    windowMin=windowMin_CT_img_HU,
                    windowMax=windowMax_CT_img_HU,
                    need_tensor_output=True, need_transform=True)

    sampler_of_airways_of_low_generation = ResetableRandomSampler(
        dataset, num_samples=min(num_samples_of_each_epoch, len(data_dict)),
        replacement=False)#replacement=False采样是否能重复
    print("dataset sample epoch ratio:",num_samples_of_each_epoch/ len(data_dict))

    dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        sampler=sampler_of_airways_of_low_generation
    )
    #LOSS
    # 定义损失函数（criterion）
    criterion = torch.nn.CrossEntropyLoss()

    for ith_epoch in range(0, max_epoch):

        len_dataset_loader = len(dataset_loader)
        sampler_of_airways_of_low_generation.set_epoch(ith_epoch)

        # loss,accuracy=train_classifier(model, dataset_loader, optimizer, criterion, device)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for ith_batch, batch in enumerate(dataset_loader):
            inputs = batch['image'].float().to(device, non_blocking=True)
            # Convert hierarchy_level to class index
            targets = batch['hierarchy_level'].long().to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            time_consumption = time.time() - start_time

            print(
                "epoch [{0}/{1}]\t"
                "batch [{2}/{3}]\t"
                "time(sec) {time:.2f}\t"
                "loss {loss:.4f}\t"
                .format(
                    ith_epoch + 1,
                    max_epoch,
                    ith_batch+1,
                    len_dataset_loader,
                    time=time_consumption,
                    loss=loss.item()
                )
            )
            writer.add_scalar('Loss/loss', loss.item(),
                              global_step=ith_epoch * len_dataset_loader + ith_batch)

            if (ith_batch + 1) % 100 == 0:
                accuracy = 100. * correct / total
                lossepcoh=running_loss / (ith_batch + 1)
                print(
                    "epoch [{0}/{1}]\t"
                    "accuracy {accuracy:.2f}%\t"
                    "loss {loss:.4f}\t"
                    .format(
                        ith_epoch + 1,
                        max_epoch,
                        accuracy=accuracy,
                        loss=lossepcoh,
                    )
                )
                writer.add_scalar('100batch/lossepoch', lossepcoh,
                                  global_step=ith_epoch * len_dataset_loader + ith_batch)
                writer.add_scalar('100batch/accuracy_cls', accuracy,
                                  global_step=ith_epoch * len_dataset_loader + ith_batch)

        if (ith_epoch + 1) % model_save_freq == 0:
            print('epoch: ' + str(ith_epoch) + ' save model')
            model.to(torch.device('cpu'))
            # Create a descriptive save path for each model state
            model_save_path = os.path.join(writer.log_dir, '400'+model_name+'_' + str(
                ith_epoch) + '.pkl')  # save_path.replace('.pkl', '_epoch_' + str(ith_epoch) + '.pkl')
            torch.save({'model_state_dict': model.state_dict()}, model_save_path)
            # torch.save(model.state_dict(), os.path.join(writer.log_dir, f'model_epoch_{epoch}.pth'))
            model.to(device)

    writer.close()
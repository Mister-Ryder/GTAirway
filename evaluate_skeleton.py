import time
from importlib import import_module
import tqdm
import skimage.measure as measure
from skimage.measure import label
import nibabel
from skimage.morphology import skeletonize_3d

import numpy as np
import torch
import copy
import pandas as pd
import SimpleITK as sitk
from PIL import Image
import pydicom
import cv2
import nibabel as nib
import os
import skimage.io as io

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from func.model_arch_model6 import SegAirwayModel
from func.model_run import get_image_and_label, get_crop_of_image_and_label_within_the_range_of_airway_foreground, \
    semantic_segment_crop_and_cat, dice_accuracy
from func.post_process import post_process, add_broken_parts_to_the_result, find_end_point_of_the_airway_centerline, \
    get_super_vox, Cluster_super_vox, delete_fragments, get_outlayer_of_a_3d_shape, get_crop_by_pixel_val, \
    fill_inner_hole
from func.detect_tree import tree_detection
from func.ulti import save_obj, load_obj, get_and_save_3d_img_for_one_case, load_one_CT_img, \
    get_df_of_centerline, get_df_of_line_of_centerline

from torch.utils.data import Dataset, DataLoader
import scipy.ndimage as ndimage
import warnings

warnings.filterwarnings('ignore')


class AirwayData(Dataset):
    def __init__(self, path, train=True):
        self.path = path
        self.path_list = os.listdir(self.path)
        self.path_list.sort()
        # self.datalen = len(self.path_list)//3
        self.datalen = len(self.path_list)
        self.train = train

    def __len__(self):
        return self.datalen

    def __getitem__(self, idx):
        # img_path = os.path.join(self.path, self.path_list[3*idx])
        # label_path = os.path.join(self.path, self.path_list[3*idx+1])
        # patient = [self.path_list[3*idx].split('_')[0]]
        img_path = os.path.join(self.path, self.path_list[idx])
        label_path = os.path.join(self.path.replace('image', 'label'), self.path_list[idx])
        patient = [self.path_list[idx].split('.')[0]]

        # img = np.load(img_path)
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img)
        # img = img[np.newaxis, :]

        # label = np.load(label_path)
        label = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label)
        # label = label[np.newaxis, :]

        # return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(label.astype(np.float32)), \
        #     torch.from_numpy(img.astype(np.float32)), patient
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(label.astype(np.float32)), torch.from_numpy(
            img.astype(np.float32)), patient


def evaluation(data_path, save_path, parsing_path,model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = SegAirwayModel(in_channels=1, out_channels=2)
    # model.to(device)
    # load_path = "checkpoint/model6-105epoch-jia140.pkl"
    # checkpoint = torch.load(load_path)
    #
    # weight_dict = {}
    # for k, v in checkpoint['model_state_dict'].items():
    #     new_k = k.replace("model.", "") if "model." in k else k
    #     weight_dict[new_k] = v
    # for k, v in checkpoint['model_state_dict'].items():
    #     new_k = k.replace("module.", "") if "module." in k else k
    #     weight_dict[new_k] = v
    # model.load_state_dict(weight_dict, strict=False)

    # 加载评估数据文件、/image/
    val_path = data_path
    dataset = AirwayData(val_path, train=False)
    val_loader_case = DataLoader(dataset, batch_size=1, shuffle=False)

    threshold = 0.5
    file_list_parse = os.listdir(parsing_path)
    file_list_parse.sort()
    file_list = os.listdir(data_path)
    file_list.sort()

    sens = []
    pres = []
    branches = []
    for i, (x, y, cb, patient) in enumerate(val_loader_case):

        x = x.cpu().detach().numpy()
        x = np.squeeze(x)
        # y = y.detach().numpy()
        y = y.squeeze().to(device)

        seg_result = semantic_segment_crop_and_cat(x, model, device,
                                                   crop_cube_size=[128, 128, 128], stride=[64, 64, 64],
                                                   windowMin=-1000, windowMax=600)
        # seg_onehot = np.array(seg_result > threshold, dtype=np.int)

        # 加载seg_result数据到GPU
        seg_onehot = torch.tensor(seg_result > threshold, dtype=torch.int, device=device)
        # seg_processed, _ = post_process(seg_onehot.cpu().detach().numpy(), threshold=threshold)
        # seg_slice_label_I, connection_dict_of_seg_I, number_of_branch_I, tree_length_I = tree_detection(seg_processed,
        #                                                                                                 search_range=2)
        # seg_processed_II = add_broken_parts_to_the_result(connection_dict_of_seg_I, seg_result, seg_processed,
        #                                                   threshold=threshold,
        #                                                   search_range=10, delta_threshold=0.05, min_threshold=0.4)

        # 加载parsing数据到GPU
        parsing = nibabel.load(os.path.join(parsing_path, file_list_parse[6 * i + 4])).get_fdata()
        parsing = torch.tensor(parsing.transpose(2, 1, 0), dtype=torch.float32, device=device)

        # Label connected components and calculate volume
        cd, num = label(seg_onehot.cpu().detach().numpy(), return_num=True, connectivity=1)#.cpu().detach().numpy()
        cd = torch.tensor(cd, dtype=torch.uint8, device=device)
        num = torch.tensor(num, dtype=torch.int, device=device)
        volume = torch.bincount(cd.view(-1)).float()  # 使用torch.bincount一次性计算所有标签的体积
        volume_sort = torch.argsort(volume, descending=True)
        if len(volume_sort) < 2:
            print("No non-background organization found in cd, skipping this sample.")
            continue
        large_cd = (cd == volume_sort[1]).to(torch.uint8)  # 获取最大体积的组织

        # Perform 3D skeletonization
        y_cpu = y.cpu().numpy()
        skeleton = skeletonize_3d(y_cpu > 0).astype(np.uint8)  # skeletonize_3d直接接受numpy数组

        # Calculate sensitivity and precision
        sen = (torch.sum(large_cd * torch.tensor(skeleton, dtype=torch.uint8, device=device)).float() / torch.sum(
            torch.tensor(skeleton, dtype=torch.uint8, device=device))).item()
        sens.append(sen)
        pre = (torch.sum(large_cd * y).float() / torch.sum(large_cd)).item()
        pres.append(pre)

        # Calculate branch statistics
        num_branch = torch.max(parsing).item()
        num_branch = int(num_branch)
        detected_num = 0
        for j in range(num_branch):
            branch_label = ((parsing == (j + 1)).to(torch.uint8)) * torch.tensor(skeleton, dtype=torch.uint8,
                                                                                 device=device)
            if (large_cd * branch_label).sum().item() / (branch_label.sum().item()+0.0001) >= 0.8:
                detected_num += 1
        branch = detected_num / num_branch
        branches.append(branch)

        # Print results
        print(file_list[i].split('.')[0], "Length: %0.4f" % (sen), "Precision: %0.4f" % (pre),
              "Branch: %0.4f" % (branch))

        '''
        # 进行保存，img和label只需要nibabel保存一次即可，后续评测可注释掉。 如果能合并，可以直接合并，x，y不再进行保存。
        save_name_x = save_path + patient[0][0].split('_')[0] + '_img' + '.nii.gz'
        save_name_p = save_path + patient[0][0].split('_')[0] + '_pred' + '.nii.gz'
        save_name_y = save_path + patient[0][0].split('_')[0] + '_label' + '.nii.gz'

        # 将图像数据转换为适当的数据类型，例如 float32
        seg_onehot = seg_onehot.cpu().detach().numpy().astype(np.float32)
        # 使用 NIfTI 格式的 header 或者明确指定数据类型（dtype）参数来创建 Nifti1Image 对象
        pred_nii = nibabel.Nifti1Image(seg_onehot, np.eye(4), dtype=np.float32)
        # pred_nii = nibabel.Nifti1Image(seg_onehot, np.eye(4))
        nibabel.save(pred_nii, save_name_p)
        print("save done")
        '''

    sen1_mean = np.mean(sens)
    sen1_std = np.std(sens)
    pre_mean = np.mean(pres)
    pre_std = np.std(pres)
    branch_mean = np.mean(branches)
    branch_std = np.std(branches)
    print("len mean: %0.4f (%0.4f), branch: %0.4f (%0.4f), pre: %0.4f (%0.4f)" % (
        sen1_mean, sen1_std, branch_mean, branch_std, pre_mean, pre_std))
    return sen1_mean, branch_mean, pre_mean


if __name__ == '__main__':
    data_path = "./BAS/test/image/"
    save_path = "./save_path/"
    parsing_path = "./BAS/test/parsing_path/"
    evaluation(data_path, save_path, parsing_path)

"""
evaluation V1

def evaluation(data_path,save_path, parsing_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SegAirwayModel(in_channels=1, out_channels=2)
    model.to(device)
    load_path = "checkpoint/checkpoint_sample.pkl"
    checkpoint = torch.load(load_path)

    weight_dict={}
    for k,v in checkpoint['model_state_dict'].items():
        new_k=k.replace("model.","") if "model."in k else k
        weight_dict[new_k]=v
    model.load_state_dict(weight_dict,strict=False)

    # 加载评估数据文件、/image/
    val_path = data_path
    dataset = AirwayData(val_path, train=False)
    val_loader_case = DataLoader(dataset, batch_size=1, shuffle=False)

    threshold = 0.5
    file_list_parse = os.listdir(parsing_path)
    file_list_parse.sort()
    file_list = os.listdir(data_path)
    file_list.sort()

    sens = []
    pres = []
    branches = []
    start_time = time.time()
    for i,(x,y,cb,patient) in enumerate(val_loader_case):

        x = x.cpu().detach().numpy()
        x = np.squeeze(x)
        y = y.detach().numpy()
        y = np.squeeze(y)

        seg_result = semantic_segment_crop_and_cat(x, model, device,
                                                   crop_cube_size=[32, 128, 128], stride=[16, 64, 64],
                                                   windowMin=-1000, windowMax=600)
        seg_onehot = np.array(seg_result > threshold, dtype=np.int)
        time_seg = time.time() - start_time
        '''
        # 这里x图像，y标签可以保存，但是nibabel的保存方向与一开始输入不一致。这里还是更改下吧。
        # 进行保存，img和label只需要nibabel保存一次即可，后续评测可注释掉。 如果能合并，可以直接合并，x，y不再进行保存。
        save_name_x = save_path + patient[0][0].split('_')[0] + '_img' + '.nii.gz'
        save_name_p = save_path + patient[0][0].split('_')[0] + '_pred' + '.nii.gz'
        save_name_y = save_path + patient[0][0].split('_')[0] + '_label' + '.nii.gz'

        # img_nii = nibabel.Nifti1Image(x, np.eye(4))
        # nibabel.save(img_nii, save_name_x)
        # label_nii = nibabel.Nifti1Image(y, np.eye(4))
        # nibabel.save(label_nii, save_name_y)
        # 将图像数据转换为适当的数据类型，例如 float32
        seg_onehot = seg_onehot.astype(np.float32)
        # 使用 NIfTI 格式的 header 或者明确指定数据类型（dtype）参数来创建 Nifti1Image 对象
        pred_nii = nibabel.Nifti1Image(seg_onehot, np.eye(4), dtype=np.float32)
        # pred_nii = nibabel.Nifti1Image(seg_onehot, np.eye(4))
        nibabel.save(pred_nii, save_name_p)
        print("save done")
        '''

        parsing = nibabel.load(os.path.join(parsing_path, file_list_parse[6 * i + 4]))  # please refer to tree_parse.py
        parsing = parsing.get_fdata()
        parsing = parsing.transpose(2, 1, 0)
        cd, num = measure.label(seg_onehot, return_num=True, connectivity=1)
        volume = np.zeros([num])
        for k in range(num):
            volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
        volume_sort = np.argsort(volume)
        # print(volume_sort)
        large_cd = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
        skeleton = skeletonize_3d(y)  # 这里这个是直接进行骨架化的。LIDC数据需要注意。
        skeleton = (skeleton > 0)
        skeleton = skeleton.astype('uint8')
        print("skeleton")

        sen = (large_cd * skeleton).sum() / skeleton.sum()
        sens.append(sen)
        pre = (large_cd * y).sum() / large_cd.sum()
        pres.append(pre)

        num_branch = parsing.max()
        num_branch = int(num_branch)
        detected_num = 0
        for j in range(num_branch):
            branch_label = ((parsing == (j + 1)).astype(np.uint8)) * skeleton
            if (large_cd * branch_label).sum() / branch_label.sum() >= 0.8:
                detected_num += 1
        branch = detected_num / num_branch
        branches.append(branch)
        #3
        print(file_list[i].split('.')[0], "Length: %0.4f" % (sen), "Precision: %0.4f" % (pre),
              "Branch: %0.4f" % (branch))
        time_evalu=time.time()-start_time-time_seg
        print("time seg:",time_seg,"time eva:",time_evalu)

    sen1_mean = np.mean(sens)
    sen1_std = np.std(sens)
    pre_mean = np.mean(pres)
    pre_std = np.std(pres)
    branch_mean = np.mean(branches)
    branch_std = np.std(branches)
    print("len mean: %0.4f (%0.4f), branch: %0.4f (%0.4f), pre: %0.4f (%0.4f)" % (
    sen1_mean, sen1_std, branch_mean, branch_std, pre_mean, pre_std))  


"""
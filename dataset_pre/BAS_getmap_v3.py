# -*- coding: utf-8 -*-
# @Time    : 2024/3/10 17:34
# @Author  : JIA
# @FileName: skeleton_getmap_atmdata.py
# @Software: PyCharm
# @Blog    ：


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
import plotly.io as pio
import plotly.express as px
import tqdm

from func.detect_tree import tree_detection
from func.ulti import save_obj, load_obj, get_and_save_3d_img_for_one_case, load_one_CT_img, \
    get_df_of_centerline, get_df_of_line_of_centerline

from func.points_to_mesh import produce_3d_obj


class Normalization_np(object):
    def __init__(self, windowMin, windowMax):
        self.name = 'ManualNormalization'
        assert isinstance(windowMax, (int, float))
        assert isinstance(windowMin, (int, float))
        self.windowMax = windowMax
        self.windowMin = windowMin

    def __call__(self, img_3d):
        img_3d_norm = np.clip(img_3d, self.windowMin, self.windowMax)
        img_3d_norm -= np.min(img_3d_norm)
        max_99_val = np.percentile(img_3d_norm, 99)
        if max_99_val > 0:
            img_3d_norm = img_3d_norm / max_99_val * 255

        return img_3d_norm


def normalize_img(img, windowMin, windowMax):
    normalization = Normalization_np(windowMin, windowMax)
    img = normalization(img)
    return img


def load_one_CT_img(img_path):
    return io.imread(img_path, plugin='simpleitk')


def get_centerline(connection_dict_of_seg_I):
    # get the centerline
    df_of_line_of_centerline = get_df_of_line_of_centerline(connection_dict_of_seg_I)


def skeletonize_3d_airway(raw_img_label_path, threshold=0.5):
    """
    input:
    raw_img_label_path: the path of the raw image
    threshold: the threshold of the model
    output:
    seg_slice_label_I: the result of the skeleton
    connection_dict_of_seg_I: the connection dict of the skeleton
    """
    raw_img_label = load_one_CT_img(raw_img_label_path)
    raw_img_label = normalize_img(raw_img_label, -1000, 600)
    raw_img_label = torch.from_numpy(raw_img_label).float()
    # threshold= 0.5
    seg_onehot = np.array(raw_img_label, dtype=np.int)

    # post process
    # seg_processed, _ = post_process(seg_onehot, threshold=threshold)
    # centerline detection
    seg_slice_label_I, connection_dict_of_seg_I, number_of_branch_I, tree_length_I, interaction_points = tree_detection(
        seg_onehot,
        search_range=2)
    print(seg_onehot.shape, number_of_branch_I, tree_length_I)

    # get the centerline
    # get_centerline(connection_dict_of_seg_I)
    # print("Finished")
    # produce 3d obj
    # produce_3d_obj(seg_processed, "./results/3dobj/labels")

    return seg_slice_label_I, connection_dict_of_seg_I, interaction_points,seg_onehot


def create_3d_distance_map(interaction_points, volume_shape, batch_size=16):
    device = torch.device("cuda")
    # Precompute meshgrid and move to GPU once
    xx, yy, zz = torch.meshgrid(torch.arange(volume_shape[0], device=device),
                                torch.arange(volume_shape[1], device=device),
                                torch.arange(volume_shape[2], device=device))
    # Flatten the grid for vectorized operations
    grid_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1).float()

    distance_map = torch.full(volume_shape, float('inf'), device=device).view(-1)

    #interaction_points = torch.tensor(interaction_points, device=device).float()

    # Process interaction points in batches
    for i in tqdm.trange(0, interaction_points.size(0), batch_size):
        batch_points = interaction_points[i:i + batch_size]
        # Compute distances from grid points to interaction points in the batch
        distances = torch.cdist(grid_points, batch_points)
        # Update the minimum distance
        min_distances, _ = distances.min(dim=1)
        distance_map = torch.min(distance_map, min_distances)

    distance_map = distance_map.view(volume_shape)
    print("Distance map calculation finished!")
    torch.cuda.empty_cache()
    return distance_map


def create_3d_gaussian_map(interaction_points, volume_shape, sigma=3, batch_size=16):
    # 初始化网格
    device = torch.device("cuda")
    xx, yy, zz = torch.meshgrid(torch.arange(volume_shape[0], device=device),
                                torch.arange(volume_shape[1], device=device),
                                torch.arange(volume_shape[2], device=device))

    gaussian_map = torch.zeros(volume_shape, device=device)

    # interaction_points = torch.tensor(interaction_points, device=device).float()

    # 批量处理以减少循环次数和利用 GPU 并行计算
    for i in range(0, len(interaction_points), batch_size):
        batch_points = interaction_points[i:i + batch_size]
        dx = (xx.unsqueeze(0) - batch_points[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3))
        dy = (yy.unsqueeze(0) - batch_points[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3))
        dz = (zz.unsqueeze(0) - batch_points[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3))

        gaussian_batch = torch.exp(-(dx ** 2 + dy ** 2 + dz ** 2) / (2 * sigma ** 2))
        gaussian_map += gaussian_batch.sum(0)

    print("Gaussian map generation finished!")
    torch.cuda.empty_cache()
    return gaussian_map


import torch
import tqdm

def create_maps(interaction_points, volume_shape, masks, sigma=3, batch_size=1):
    device = torch.device("cuda")
    # 生成网格点
    xx, yy, zz = torch.meshgrid(
        torch.arange(volume_shape[0], device=device),
        torch.arange(volume_shape[1], device=device),
        torch.arange(volume_shape[2], device=device),
        indexing='ij'
    )
    grid_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1).float()

    # 初始化距离图和高斯图
    distance_map = torch.full(volume_shape, float('inf'), device=device).view(-1)
    gaussian_map = torch.zeros(volume_shape, device=device)

    # 找到所有mask边界的点
    mask_edge_points = torch.stack(torch.nonzero(masks == 0, as_tuple=True), dim=-1).float()

    # 处理每个批次
    for i in tqdm.trange(0, interaction_points.size(0), batch_size):
        batch_points = interaction_points[i:i + batch_size].to(device)
        # 距离图计算
        distances = torch.cdist(grid_points, batch_points)
        min_distances, _ = distances.min(dim=1)
        distance_map = torch.min(distance_map, min_distances)

        # 高斯图计算
        for j in range(batch_points.size(0)):
            # 计算到mask边界的最短距离
            distances_to_edge = torch.cdist(mask_edge_points.unsqueeze(0), batch_points[j:j+1]).squeeze()
            min_distance_to_edge = distances_to_edge.min()
            sigma_adjusted = sigma * (min_distance_to_edge / sigma)  # 根据到边界的距离调整sigma

            dx = xx - batch_points[j, 0]
            dy = yy - batch_points[j, 1]
            dz = zz - batch_points[j, 2]
            gaussian_contribution = torch.exp(-(dx ** 2 + dy ** 2 + dz ** 2) / (2 * sigma_adjusted ** 2))
            gaussian_map += gaussian_contribution * masks  # 仅在mask内部累加高斯贡献

    # 重塑距离图
    distance_map = distance_map.view(volume_shape)

    torch.cuda.empty_cache()
    return distance_map, gaussian_map



if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    #dir=["/TrainBatch1","/TrainBatch2","/imagesVal"]
    #for name in dir:
    device = torch.device("cuda")
    raw_file_path = "/home/ubuntu/shared-data/BAS/train" #train

    raw_label_file_path = raw_file_path + "/label"
    raw_label_skeleton_file_path = raw_file_path + "/train_label_skeleton3"
    raw_mapdistance_file_path = raw_file_path + "/train_map_distance3"
    raw_mapgaussian_file_path = raw_file_path + "/train_map_gaussian3"
    if not os.path.exists(raw_label_skeleton_file_path):
        os.makedirs(raw_label_skeleton_file_path)
    if not os.path.exists(raw_mapdistance_file_path):
        os.makedirs(raw_mapdistance_file_path)
    if not os.path.exists(raw_mapgaussian_file_path):
        os.makedirs(raw_mapgaussian_file_path)

    # get the list of the raw label
    label_case_dict = dict()
    label_case_names = os.listdir(raw_label_file_path)
    label_case_names.sort()  # 排序
    # label_case_names.reverse()  # 倒序排列

    for case_name in label_case_names:
        label_case_dict[case_name] = raw_label_file_path + "/" + case_name
        # print(case_name)
        print(label_case_dict[case_name])
        # get the skeleton of the label
        seg_slice_label_I, connection_dict_of_seg_I, interaction_points,label_mask= skeletonize_3d_airway(label_case_dict[case_name], threshold=0.5)
        sitk.WriteImage(sitk.GetImageFromArray(seg_slice_label_I.astype(np.int16)),
                        raw_label_skeleton_file_path + "/" + case_name)  # label_skeleton_"results/label_seleten.nii.gz"
        print(case_name + " Skeletonlize Done!")

        volume_shape = interaction_points["shape"]
        interaction_points = interaction_points["loc"]
        interaction_points = torch.tensor(interaction_points, device=device).float()

        label_mask=torch.tensor(label_mask,device=device)
        distance_map,gaussian_map =create_maps(interaction_points, volume_shape,label_mask)

        #distance_map = create_3d_distance_map(interaction_points, volume_shape)
        sitk.WriteImage(sitk.GetImageFromArray(distance_map.cpu().numpy().astype(np.int16)),
                        raw_mapdistance_file_path + "/" + case_name)

        #gaussian_map = create_3d_gaussian_map(interaction_points, volume_shape, sigma=3)
        sitk.WriteImage(sitk.GetImageFromArray(gaussian_map.cpu().numpy().astype(np.int16)),
                        raw_mapgaussian_file_path + "/" + case_name)
        print(case_name + " Distance and Gaussian Map Done!\n")
        torch.cuda.empty_cache()



"""
v1
def create_3d_gaussian_map(interaction_points,volume_shape, sigma=3):
    xx, yy, zz = torch.meshgrid(torch.arange(volume_shape[0]), torch.arange(volume_shape[1]),
                                torch.arange(volume_shape[2]))  # 生成三维网格
    xx = xx.to(device)
    yy = yy.to(device)
    zz = zz.to(device)

    gaussian_map = torch.zeros(volume_shape, device="cuda")
    # 循环取interaction_points中的前1000个点并加一个进度条
    # for point in tqdm.tqdm(interaction_points[:1000]):
    for point in tqdm.tqdm(interaction_points):
        point = torch.tensor(point,device=torch.device("cuda"))
        gaussian_map += torch.exp(
            -((xx - point[0]) ** 2 + (yy - point[1]) ** 2 + (zz - point[2]) ** 2) / (2 * sigma ** 2))

    print("gaussian_map finish!")
    return gaussian_map
    

def create_3d_distance_map(interaction_points, volume_shape):

    xx, yy, zz = torch.meshgrid(torch.arange(volume_shape[0]), torch.arange(volume_shape[1]),
                                torch.arange(volume_shape[2]))  # 生成三维网格
    xx = xx.to(device)
    yy = yy.to(device)
    zz = zz.to(device)
    distance_map = torch.full(volume_shape, float('inf'), device="cuda")  # 初始化一个全为inf的tensor
    # 我们需要给for循环加一个进度条，这样可以看到进度
    # for point in tqdm.tqdm(interaction_points[:1000]):
    for point in tqdm.tqdm(interaction_points):
        point = torch.tensor(point,device=torch.device("cuda"))
        distance = torch.sqrt((xx - point[0]) ** 2 + (yy - point[1]) ** 2 + (zz - point[2]) ** 2)  # 计算每个点到交互点的距离
        distance_map = torch.min(distance_map, distance)  # 取最小值

    print("distance_map finish!")
    return distance_map
"""
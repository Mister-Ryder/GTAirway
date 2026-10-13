# dataset load and transform
import numpy as np
import skimage.io as io
import random
import h5py

from torch.utils.data import Dataset
from torch import from_numpy as from_numpy
from torchvision import transforms
import torch
import torchio as tio
import nibabel
import scipy.ndimage as ndimage

np.random.seed(114514)

def central_crop(sample, label, dist, crop_size):
    origin_size = sample.shape
    crop_size = np.array(crop_size)
    start = (origin_size - crop_size) // 2
    sample = sample[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
             start[2]:(start[2] + crop_size[2])]
    label = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1], start[2]:start[2] + crop_size[2]]
    dist = dist[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1], start[2]:start[2] + crop_size[2]]
    return sample, label, dist

def skeleton_sample(img, label, gaussian, loc, cube_size):
    origin_size = img.shape
    # print("origin_size", origin_size)
    crop_size = np.array([cube_size, cube_size, cube_size])
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    gaussian_crop = gaussian[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, gaussian_crop

def small_airway_sample(img, label, gaussian, loc, cube_size):
    origin_size = img.shape
    # print("origin_size",origin_size)
    crop_size = [cube_size, cube_size, cube_size]
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    gaussian_crop = gaussian[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, gaussian_crop

def random_sample(img, label, gaussian, cube_size):
    origin_size = img.shape
    crop_size = [cube_size, cube_size, cube_size]
    start = [np.random.randint(0, origin_size[0] - crop_size[0]), np.random.randint(0, origin_size[1] - crop_size[1]),
             np.random.randint(0, origin_size[2] - crop_size[2])]
    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    gaussian_crop = gaussian[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    return img_crop, label_crop, gaussian_crop

class Random3DCrop_np(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple)), 'Attention: random 3D crop output size: an int or a tuple (length:3)'
        if isinstance(output_size, int):
            self.output_size=(output_size, output_size, output_size)
        else:
            assert len(output_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
            self.output_size=output_size
        
    def random_crop_start_point(self, input_size):
        assert len(input_size)==3, 'Attention: random 3D crop output size: a tuple (length:3)'
        d, h, w=input_size
        d_new, h_new, w_new=self.output_size

        d_start = np.random.randint(0, d - d_new + 1)
        h_start = np.random.randint(0, h - h_new + 1)
        w_start = np.random.randint(0, w - w_new + 1)

        # Check if the randomly chosen start point allows for the full crop
        # If not, adjust the start point to ensure the crop fits within the dimensions
        if d_start + d_new > d:
            d_start = d - d_new
        if h_start + h_new > h:
            h_start = h - h_new
        if w_start + w_new > w:
            w_start = w - w_new
        
        return d_start, h_start, w_start

    def __call__(self, img_3d, start_points=None):
        img_3d=np.array(img_3d)
        d, h, w=img_3d.shape
        d_new, h_new, w_new=self.output_size
        if start_points == None:
            start_points = self.random_crop_start_point(img_3d.shape)

        d_start, h_start, w_start = start_points
        d_end = min(d_start+d_new, d)
        h_end = min(h_start+h_new, h)
        w_end = min(w_start+w_new, w)

        crop=img_3d[d_start:d_end, h_start:h_end, w_start:w_end]

        return crop

class Normalization_np(object):
    def __init__(self, windowMin, windowMax):
        self.name = 'ManualNormalization'
        assert isinstance(windowMax, (int,float))
        assert isinstance(windowMin, (int,float))
        self.windowMax = windowMax
        self.windowMin = windowMin
    
    def __call__(self, img_3d):
        img_3d_norm = np.clip(img_3d, self.windowMin, self.windowMax)
        img_3d_norm-=np.min(img_3d_norm)
        max_99_val=np.percentile(img_3d_norm, 99)
        if max_99_val>0:
            img_3d_norm = img_3d_norm/max_99_val*255
        return img_3d_norm

import numpy as np
import torch
import os
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
import SimpleITK as sitk

class semi_airway_dataset(Dataset):
    def __init__(self, file_list, data_root,num_of_samples = None):
        # each item of data_dict is {name:{"image": img path, "label": label path}}
        self.file_list = file_list
        self.root = data_root
        self.para = {}
        self.set_para()
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):

        return self.get(idx, file_format=self.para["file_format"],\
            crop_size=self.para["crop_size"],\
                windowMin=self.para["windowMin"],\
                    windowMax=self.para["windowMax"],\
                        need_tensor_output=self.para["need_tensor_output"],\
                            need_transform=self.para["need_transform"])

    def set_para(self, file_format='.nii.gz', crop_size=(128,128,128), windowMin=-1000, windowMax=150,\
        need_tensor_output=True, need_transform=True):
        self.para["file_format"] = file_format
        self.para["crop_size"] = crop_size
        self.para["windowMin"] = windowMin
        self.para["windowMax"] = windowMax
        self.para["need_tensor_output"] = need_tensor_output
        self.para["need_transform"] = need_transform

    def crop(self, img, label, gaussian, skeleton,cube_size=128):
        # img_crops, label_crops, gaussian_crops = [], [], []
        # dis = ndimage.distance_transform_edt(label)
        # loc_small = np.where((dis * skeleton) < 2)
        loc_skeleton0=np.where(skeleton)
        loc_skeleton = np.where(skeleton * label)
        loc_skeleton2= np.where(skeleton * (1-label))
        print(loc_skeleton)
        # cube_size = cube_size

        # for i in range(self.batch_size):
        # p = np.random.random()
        # skeleton hard sample mining
        # if p > 0.5:  # 0.5
        # img_crop, label_crop, gaussian_crop = skeleton_sample(img, label, gaussian, loc_skeleton, cube_size)
        # sample on small airway
        # else :  # 1-0.5
        #     img_crop, label_crop, gaussian_crop = small_airway_sample(img, label, gaussian, loc_small, cube_size)

        # random sampling
        # else:
        img_crop, label_crop, gaussian_crop = random_sample(img, label, gaussian,cube_size)
        # img_crops.append(img_crop)
        # label_crops.append(label_crop)
        # gaussian_crops.append(gaussian_crop)

        return img_crop, label_crop, gaussian_crop

    def get(self, idx, file_format='.nii.gz', crop_size=(128,128,128), windowMin=-1000, windowMax=150,\
        need_tensor_output=True, need_transform=True):

        normalization=Normalization_np(windowMin, windowMax)

        name = self.file_list[idx]
        # print(name)
        if file_format == '.nii.gz':
            img = sitk.ReadImage(os.path.join(self.root, 'imagesTr', name + '.nii.gz'))
            img = sitk.GetArrayFromImage(img)
            img =np.array(img, float)
        #     label = sitk.ReadImage(os.path.join(self.root, 'labelsTr', name + '.nii.gz'))
        #     label = sitk.GetArrayFromImage(label)
        #     label = np.array(label, float)
        #     gaussian = sitk.ReadImage(os.path.join(self.root, 'maps_gaussianTr', name + '.nii.gz'))
        #     gaussian = sitk.GetArrayFromImage(gaussian)
        #     gaussian = np.array(gaussian, float)
        #     skeleton = sitk.ReadImage(os.path.join(self.root, 'labels_skeletonTr', name + '.nii.gz'))
        #     skeleton = sitk.GetArrayFromImage(skeleton)
        #     skeleton= np.array(skeleton, float)
        #
        # assert img.shape == label.shape
        # origin_size = img.shape
        # print(origin_size,name)
        random3dcrop = Random3DCrop_np(crop_size)

        # p = np.random.random()
        # if p > 0.75: #0.25的比例
        start_points=random3dcrop.random_crop_start_point(img.shape)
        raw_img_crop=random3dcrop(img, start_points=start_points)
        # label_img_crop=random3dcrop(label, start_points=start_points)
        # gaussian_img_crop = random3dcrop(gaussian, start_points=start_points)
        #print(raw_img_crop.shape,label_img_crop.shape,gaussian_img_crop.shape)
        # else : #0.75的比例
        #     raw_img_crop, label_img_crop, gaussian_img_crop = self.crop(img, label, gaussian, skeleton, cube_size=crop_size)
            # print(raw_img_crop.shape, label_img_crop.shape, gaussian_img_crop.shape)

        raw_img_crop = normalization(raw_img_crop)

        raw_img_crop = np.expand_dims(raw_img_crop, axis=0)
        # label_img_crop = np.expand_dims(label_img_crop, axis=0)
        # skeleton_img_crop = np.expand_dims(skeleton_img_crop, axis=0)
        # distance_img_crop = np.expand_dims(distance_img_crop, axis=0)
        # gaussian_img_crop = np.expand_dims(gaussian_img_crop, axis=0)
        # skeleton_img_crop=random3dcrop(np.array(img, float), start_points=start_points)
        # distance_img_crop = random3dcrop(np.array(img, float), start_points=start_points)

        output = {"image": raw_img_crop}# "label": label_img_crop, "gaussian": gaussian_img_crop}
        # else:
        #     output = {"image": raw_img_crop, "label": label_img_crop}

        if need_tensor_output:
            output = self.to_tensor(output)
            if need_transform:
                output = self.transform_the_tensor(output, prob=0.5)

        return output

    def to_tensor(self, images):
        for item in images.keys():
            images[item]=from_numpy(images[item]).float()
        return images
    
    def transform_the_tensor(self, image_tensors, prob=0.5):
        dict_imgs_tio={}
        
        for item in image_tensors.keys():
            dict_imgs_tio[item]=tio.ScalarImage(tensor=image_tensors[item])
        subject_all_imgs = tio.Subject(dict_imgs_tio)
        transform_shape = tio.Compose([
            tio.RandomFlip(axes = int(np.random.randint(3, size=1)[0]), p=prob),tio.RandomAffine(p=prob)])
        transformed_subject_all_imgs = transform_shape(subject_all_imgs)
        transform_val = tio.Compose([
            tio.RandomBlur(p=prob),\
                tio.RandomNoise(p=prob),\
                    tio.RandomMotion(p=prob),\
                        tio.RandomBiasField(p=prob),\
                            tio.RandomSpike(p=prob),\
                                tio.RandomGhosting(p=prob)])
        transformed_subject_all_imgs['image'] = transform_val(transformed_subject_all_imgs['image'])
        
        for item in subject_all_imgs.keys():
            image_tensors[item] = transformed_subject_all_imgs[item].data
        
        return image_tensors
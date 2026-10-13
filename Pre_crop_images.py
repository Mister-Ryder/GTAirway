#!/usr/bin/env python
# coding: utf-8

# # Generate pre-cropped EXACT09 and LIDC-IDRI for faster training


import os
import numpy as np
import skimage.io as io
import SimpleITK as sitk


# build the raw_data_dict for train
# EXACT09 train data

raw_data_dict = dict()

EXACT09_file_path = "/mnt/e/00-Data/Airway/EXACT09/TEMP"#"/data/Airway/EXACT09_3D"
EXACT09_train_raw_path = EXACT09_file_path+"/train"
EXACT09_train_label_path = EXACT09_file_path+"/train_label"

EXACT09_raw_names = os.listdir(EXACT09_train_raw_path)
EXACT09_raw_names.sort()

EXACT09_label_names = os.listdir(EXACT09_train_label_path)
EXACT09_label_names.sort()

case_names = []

for case in EXACT09_raw_names:
    temp = case.split(".")[0]
    case_names.append(temp)
    raw_data_dict["EXACT09_"+temp]={}
    raw_data_dict["EXACT09_"+temp]["image"]=EXACT09_train_raw_path+"/"+case

for case in EXACT09_label_names:
    temp = case.split("_")[0]
    if temp in case_names:
        raw_data_dict["EXACT09_"+temp]["label"]=EXACT09_train_label_path+"/"+case

EXACT09_data_dict = raw_data_dict

#print(EXACT09_data_dict)


def crop_one_3d_img(input_img, crop_cube_size, stride):
    # input_img: 3d matrix, numpy.array

    # crop the whole image 如果不进行裁剪，直接返回原图
    crop_cube_size = input_img.shape
    stride = input_img.shape
    ########################################

    assert isinstance(crop_cube_size, (int, tuple))
    if isinstance(crop_cube_size, int):
        crop_cube_size=np.array([crop_cube_size, crop_cube_size, crop_cube_size])
    else:
        assert len(crop_cube_size)==3
    
    crop_cube_size = (min(crop_cube_size[0], input_img.shape[0]),
                      min(crop_cube_size[1], input_img.shape[1]),
                      min(crop_cube_size[2], input_img.shape[2]))
    
    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
        stride=np.array([stride, stride, stride])
    else:
        assert len(stride)==3

    # 获取图像形状并计算总裁剪块数量
    img_shape=input_img.shape
    
    total=len(np.arange(0, img_shape[0], stride[0]))*len(np.arange(0, img_shape[1], stride[1]))*len(np.arange(0, img_shape[2], stride[2]))
    
    count=0
    
    crop_list = []

    # 遍历图像，按照给定的步长和裁剪块大小进行分割
    for i in np.arange(0, img_shape[0], stride[0]):
        for j in np.arange(0, img_shape[1], stride[1]):
            for k in np.arange(0, img_shape[2], stride[2]):

                # 计算裁剪块的起始和结束索引
                print('crop one 3d img progress : '+str(np.int(count/total*100))+'%', end='\r')
                if i+crop_cube_size[0]<=img_shape[0]:
                    x_start_input=i
                    x_end_input=i+crop_cube_size[0]
                    x_start_output=i
                    x_end_output=i+stride[0]
                else:
                    x_start_input=img_shape[0]-crop_cube_size[0]
                    x_end_input=img_shape[0]
                    x_start_output=i
                    x_end_output=img_shape[0]
                
                if j+crop_cube_size[1]<=img_shape[1]:
                    y_start_input=j
                    y_end_input=j+crop_cube_size[1]
                    y_start_output=j
                    y_end_output=j+stride[1]
                else:
                    y_start_input=img_shape[1]-crop_cube_size[1]
                    y_end_input=img_shape[1]
                    y_start_output=j
                    y_end_output=img_shape[1]
                
                if k+crop_cube_size[2]<=img_shape[2]:
                    z_start_input=k
                    z_end_input=k+crop_cube_size[2]
                    z_start_output=k
                    z_end_output=k+stride[2]
                else:
                    z_start_input=img_shape[2]-crop_cube_size[2]
                    z_end_input=img_shape[2]
                    z_start_output=k
                    z_end_output=img_shape[2]

                # 裁剪图像块并添加到列表中
                crop_temp=input_img[x_start_input:x_end_input, y_start_input:y_end_input, z_start_input:z_end_input]
                crop_list.append(np.array(crop_temp, dtype=np.float))
                
                count=count+1
                
    return crop_list

# #### pre crop EXACT09

crop_cube_size=(128,128,128)
stride=(64,64,64)

# -----INPUT-----
output_file_path = "Precrop_dataset_for_EXACT09_no_crop"

if not os.path.exists(output_file_path+"/image/"):
    os.makedirs(output_file_path+"/image/")

if not os.path.exists(output_file_path+"/label/"):
    os.makedirs(output_file_path+"/label/")

raw_data_dict = EXACT09_data_dict
# -----END-----

for i, case in enumerate(raw_data_dict.keys()):
    raw_img = io.imread(raw_data_dict[case]["image"], plugin='simpleitk')
    label_img = io.imread(raw_data_dict[case]["label"], plugin='simpleitk')
    
    raw_img_crop_list = crop_one_3d_img(raw_img, crop_cube_size=crop_cube_size, stride=stride)
    label_img_crop_list = crop_one_3d_img(label_img, crop_cube_size=crop_cube_size, stride=stride)
    
    assert len(raw_img_crop_list)==len(label_img_crop_list)
    
    for idx in range(len(raw_img_crop_list)):
        print("progress: "+str(idx)+"th crop | "+str(i)+"th 3d img: "+str(case), end="\r")
        
        #sitk.WriteImage(sitk.GetImageFromArray(raw_img_crop_list[idx]), output_file_path+"/image/"+case+"_"+str(idx)+".nii.gz")
        #sitk.WriteImage(sitk.GetImageFromArray(label_img_crop_list[idx]), output_file_path+"/label/"+case+"_"+str(idx)+".nii.gz")
        
        np.save(output_file_path+"/image/"+case+"_"+str(idx)+".npy", raw_img_crop_list[idx])
        np.save(output_file_path+"/label/"+case+"_"+str(idx)+".npy", label_img_crop_list[idx])




"""
# build the raw_data_dict for train
# LIDC-IDRI train data

raw_data_dict = dict()

# LIDC-IDRI data
LIDC_IDRI_file_path = "/data/Airway/LIDC-IDRI_3D/annotated_data"
LIDC_IDRI_raw_path = LIDC_IDRI_file_path+"/image"
LIDC_IDRI_label_path = LIDC_IDRI_file_path+"/label"

LIDC_IDRI_raw_names = os.listdir(LIDC_IDRI_raw_path)
LIDC_IDRI_raw_names.sort()

LIDC_IDRI_label_names = os.listdir(LIDC_IDRI_label_path)
LIDC_IDRI_label_names.sort()

case_names = []

for case in LIDC_IDRI_raw_names:
    temp = case.split(".")[0]
    #print(temp)
    case_names.append(temp)
    raw_data_dict["LIDC_IDRI_"+temp]={}
    raw_data_dict["LIDC_IDRI_"+temp]["image"]=LIDC_IDRI_raw_path+"/"+case

for case in LIDC_IDRI_label_names:
    temp = case.split(".")[0]
    #print(temp)
    if temp in case_names:
        raw_data_dict["LIDC_IDRI_"+temp]["label"]=LIDC_IDRI_label_path+"/"+case

LIDC_IDRI_data_dict = raw_data_dict


print(LIDC_IDRI_data_dict)
       

# #### pre crop LIDC-IDRI

crop_cube_size = (256, 256, 256)
stride = (128, 128, 128)

# -----INPUT-----
output_file_path = "Precrop_dataset_for_LIDC-IDRI"

if not os.path.exists(output_file_path + "/image/"):
    os.makedirs(output_file_path + "/image/")

if not os.path.exists(output_file_path + "/label/"):
    os.makedirs(output_file_path + "/label/")

raw_data_dict = LIDC_IDRI_data_dict
# -----END-----

for i, case in enumerate(raw_data_dict.keys()):
    raw_img = io.imread(raw_data_dict[case]["image"], plugin='simpleitk')
    label_img = io.imread(raw_data_dict[case]["label"], plugin='simpleitk')

    raw_img_crop_list = crop_one_3d_img(raw_img, crop_cube_size=crop_cube_size, stride=stride)
    label_img_crop_list = crop_one_3d_img(label_img, crop_cube_size=crop_cube_size, stride=stride)

    assert len(raw_img_crop_list) == len(label_img_crop_list)

    for idx in range(len(raw_img_crop_list)):
        print("progress: " + str(idx) + "th crop | " + str(i) + "th 3d img: " + str(case), end="\r")

        # sitk.WriteImage(sitk.GetImageFromArray(raw_img_crop_list[idx]), output_file_path+"/image/"+case+"_"+str(idx)+".nii.gz")
        # sitk.WriteImage(sitk.GetImageFromArray(label_img_crop_list[idx]), output_file_path+"/label/"+case+"_"+str(idx)+".nii.gz")

        np.save(output_file_path + "/image/" + case + "_" + str(idx) + ".npy", raw_img_crop_list[idx])
        np.save(output_file_path + "/label/" + case + "_" + str(idx) + ".npy", label_img_crop_list[idx])
  
  
        
"""
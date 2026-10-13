# -*- coding: utf-8 -*-
# @Time    : 2024/3/27 0:49
# @Author  : JIA
# @FileName: TEST.py
# @Software: PyCharm
# @Blog    ：
import SimpleITK as sitk
import os
import tqdm

file = open('info_BAS.txt', "w")

def info(image_path):
    image = sitk.ReadImage(image_path)
    image_size = image.GetSize()
    image_spacing = image.GetSpacing()
    image_direction = image.GetDirection()
    image_name=image_path.split('/')[-1]
    print(f"Image Path: {image_path}, Image Size: {image_size}, Spacing: {image_spacing}, Direction: {image_direction}")
    print(f"Image Path: {image_path}, Image Size: {image_size}, Spacing: {image_spacing}, Direction: {image_direction}",file=file)

raw_file_path='/home/ubuntu/shared-data/BAS/'
train_image='train/image'
train_label='train/label'
test_image='test/image'
test_label='test/label'
case=[train_image,train_label,test_image,test_label]
# get the list of the raw label
case_dict = dict()

for i in case:
    case_names = os.listdir(raw_file_path+i)
    case_names.sort()
    for case_name in case_names:
        case_dict[case_name] = raw_file_path + i + "/" + case_name
        info(case_dict[case_name])

file.close()

file = open('info_EXACT09.txt', "w")

raw_file_path='/home/ubuntu/shared-data/EXACT09/'
train_image='train'
train_label='train_label'
test_image='test'
case=[train_image,train_label,test_image]
# get the list of the raw label
case_dict = dict()

for i in case:
    case_names = os.listdir(raw_file_path+i)
    case_names.sort()
    for case_name in case_names:
        case_dict[case_name] = raw_file_path + i + "/" + case_name
        info(case_dict[case_name])

file.close()

file = open('info_LIDC-IDRI.txt', "w")

raw_file_path='/home/ubuntu/shared-data/LIDC-IDRI_3D/annotated_data/'
train_image='image'
train_label='label'
case=[train_image,train_label]
# get the list of the raw label
case_dict = dict()

for i in case:
    case_names = os.listdir(raw_file_path+i)
    case_names.sort()
    for case_name in case_names:
        case_dict[case_name] = raw_file_path + i + "/" + case_name
        info(case_dict[case_name])

file.close()
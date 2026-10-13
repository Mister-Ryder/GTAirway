#!/usr/bin/env python
# coding: utf-8

# # Pre-process LIDC-IDR


import numpy as np
import os
import SimpleITK as sitk
from PIL import Image
import pydicom
import cv2
import nibabel as nib
import pydicom
import tqdm

def loadFile(filename):
    ds = sitk.ReadImage(filename)
    #pydicom.dcmread(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape
    #print("frame_num, width, height: "+str((frame_num, width, height)))
    return img_array, frame_num, width, height

'''
def loadFileInformation(filename):
    information = {}
    ds = pydicom.read_file(filename)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['Manufacturer'] = ds.Manufacturer
    return information
'''

def get_3d_img_for_one_case(img_path_list, img_format="dcm"):
    img_3d=[]
    for idx, img_path in enumerate(img_path_list):
        #print("progress: "+str(idx/len(img_path_list))+"; "+str(img_path), end="\r")
        img_slice, frame_num, _, _ = loadFile(img_path)
        assert frame_num==1
        img_3d.append(img_slice)
    img_3d=np.array(img_3d)
    return img_3d.reshape(img_3d.shape[0], img_3d.shape[2], img_3d.shape[3])


# ### for all LIDC-IDRI images

# the path to LIDC-IDRI raw images

LIDC_IDRI_raw_path = "/mnt/e/00-Data/manifest-1600709154662/LIDC-IDRI"#"/data/Airway/LIDC-IDRI"

# The file structure should be like this\
# \
# /data/Airway/LIDC-IDRI\
#     /LIDC-IDRI-0001\
#         /1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\
#             /1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192\
#                 /1-001.dcm\
#                 /1-002.dcm\
#                 ...\
#     /LIDC-IDRI-0002\
#     ...

LIDC_IDRI_raw_img_dict = {}
img_names = os.listdir(LIDC_IDRI_raw_path)
img_names.sort()

path_to_a_case = ""

def find_imgs(input_path):
    global path_to_a_case
    items = os.listdir(input_path)
    items.sort()
    #print("There are "+str(items)+" in "+str(input_path))
    All_file_flag = True
    for item in items:
        if os.path.isdir(input_path+"/"+item):
            All_file_flag = False
            break
    if All_file_flag and len(items)>10:
        #print("we get "+str(input_path))
        path_to_a_case = input_path
    else:
        for item in items:
            if os.path.isdir(input_path+"/"+item):
                #print("open filefloder: "+str(input_path+"/"+item))
                find_imgs(input_path+"/"+item)

# 加一个进度条
for idx, img_name in tqdm.tqdm(enumerate(img_names)):
    #print(idx/len(img_names), end="\r")
    find_imgs(LIDC_IDRI_raw_path+"/"+img_name)
    slice_names = os.listdir(path_to_a_case)
    slice_names.sort()
    LIDC_IDRI_raw_img_dict[img_name]=[]
    for slice_name in slice_names:
        if slice_name.split(".")[1]=="dcm":
            LIDC_IDRI_raw_img_dict[img_name].append(path_to_a_case+"/"+slice_name)


# ### Some functions

# set output path

output_image_path = "/mnt/e/00-Data/manifest-1600709154662/LIDC-IDRI_TEMP" #"LIDC-IDRI"

if not os.path.exists(output_image_path):
    os.mkdir(output_image_path)
# 加一个进度条
for case in tqdm.tqdm(reversed(LIDC_IDRI_raw_img_dict.keys())):
#or case in LIDC_IDRI_raw_img_dict.keys():
    img_3d = get_3d_img_for_one_case(LIDC_IDRI_raw_img_dict[case])
    sitk.WriteImage(sitk.GetImageFromArray(img_3d),
                    output_image_path+"/"+case+".nii.gz")


# ### for the labeled LIDC-IDRI images

LIDC_IDRI_anno_path = "/mnt/e/00-Data/manifest-1600709154662/LIDC-IDRI_annotation"

anno_names = os.listdir(LIDC_IDRI_anno_path)
for idx, name in enumerate(anno_names):
    anno_names[idx]=name.split(".nii.gz")[0]


LIDC_IDRI_all_raw_img_dict = {}
for case in os.listdir(LIDC_IDRI_raw_path):
    for name_1 in os.listdir(LIDC_IDRI_raw_path+"/"+case):
        for name_2 in os.listdir(LIDC_IDRI_raw_path+"/"+case+"/"+name_1):
            img_names = os.listdir(LIDC_IDRI_raw_path+"/"+case+"/"+name_1+"/"+name_2)
            img_names.sort()
            if len(img_names)>10:
                LIDC_IDRI_all_raw_img_dict[case.split("-")[2]] = []
                for slice_name in img_names:
                    if slice_name.split(".")[1]=="dcm":
                        LIDC_IDRI_all_raw_img_dict[case.split("-")[2]].append(LIDC_IDRI_raw_path+"/"+case+"/"+name_1+"/"+name_2+"/"+slice_name)


LIDC_IDRI_annotated = {}
LIDC_IDRI_annotation_path = LIDC_IDRI_anno_path

for case in os.listdir(LIDC_IDRI_raw_path):
    for name_1 in os.listdir(LIDC_IDRI_raw_path+"/"+case):
        for name_2 in os.listdir(LIDC_IDRI_raw_path+"/"+case+"/"+name_1):
            if name_2 in anno_names:
                print(LIDC_IDRI_raw_path+"/"+case+"/"+name_1+"/"+name_2)
                img_names = os.listdir(LIDC_IDRI_raw_path+"/"+case+"/"+name_1+"/"+name_2)
                img_names.sort()
                LIDC_IDRI_annotated[case.split("-")[2]] = {}
                LIDC_IDRI_annotated[case.split("-")[2]]["image"] = []
                LIDC_IDRI_annotated[case.split("-")[2]]["label"] = LIDC_IDRI_annotation_path+"/"+name_2+".nii.gz"
                for slice_name in img_names:
                    if slice_name.split(".")[1]=="dcm":
                        LIDC_IDRI_annotated[case.split("-")[2]]["image"].append(LIDC_IDRI_raw_path+"/"+case+"/"+name_1+"/"+name_2+"/"+slice_name)


### show the info of dcm
for case in LIDC_IDRI_annotated.keys():
    print(case, end="\n")
    dicom_file = pydicom.dcmread(LIDC_IDRI_annotated[case]["image"][0])
    keys = list(dicom_file.keys())
    for idx, key in enumerate(keys):
        if str(key) == "(0028, 0030)":
            i = idx
            break
    print(dicom_file[list(dicom_file.keys())[i]])
    print("----------")


# Save the images and annotations

output_image_path = "/mnt/e/00-Data/manifest-1600709154662/LIDC-IDRI_TEMP/image"
output_label_path = "/mnt/e/00-Data/manifest-1600709154662/LIDC-IDRI_TEMP/label"


if not os.path.exists(output_image_path):
    os.makedirs(output_image_path)

if not os.path.exists(output_label_path):
    os.makedirs(output_label_path)

# 加一个进度条
for case in tqdm.tqdm(LIDC_IDRI_annotated.keys()):
#for case in LIDC_IDRI_annotated.keys():
    img_3d = get_3d_img_for_one_case(LIDC_IDRI_annotated[case]["image"])
    sitk.WriteImage(sitk.GetImageFromArray(img_3d),
                    output_image_path+"/"+case+".nii.gz")
    img_label_arr, _, _, _ = loadFile(LIDC_IDRI_annotated[case]["label"])
    sitk.WriteImage(sitk.GetImageFromArray(img_label_arr),
                    output_label_path+"/"+case+'.nii.gz')


print("Show the case names: "+str(LIDC_IDRI_raw_img_dict.keys()))


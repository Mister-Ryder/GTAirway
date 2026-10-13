import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
import pydicom
import cv2
import nibabel as nib



raw_train_file_path = "/mnt/e/00-Data/Airway/EXACT09/train_data/Training/"
raw_test_file_path = "/mnt/e/00-Data/Airway/EXACT09/test_data/Testing/"

train_case_dict = dict()
test_case_dict = dict()

train_case_names=os.listdir(raw_train_file_path)
train_case_names.sort()

test_case_names=os.listdir(raw_test_file_path)
test_case_names.sort()

# get the image paths of each case
for case_name in train_case_names:
    imgs=os.listdir(raw_train_file_path+"/"+case_name)
    imgs.sort()
    img_path_list = []
    for img in imgs:
        img_path_list.append(raw_train_file_path+"/"+case_name+"/"+img)
    train_case_dict[case_name]=img_path_list

for case_name in test_case_names:
    imgs=os.listdir(raw_test_file_path+"/"+case_name)
    imgs.sort()
    img_path_list = []
    for img in imgs:
        img_path_list.append(raw_test_file_path+"/"+case_name+"/"+img)
    test_case_dict[case_name]=img_path_list #


for case in test_case_dict.keys():
    '''
    输出每个case的第一个图像的(0028, 0030)标签，即像素间距
    '''
    print(case, end="\n")
    dicom_file = pydicom.dcmread(test_case_dict[case][0]) # 读取第一个图像
    keys = list(dicom_file.keys()) # 获取所有的标签
    for idx, key in enumerate(keys):
        # 找到(0028, 0030)标签
        if str(key) == "(0028, 0030)":
            i = idx
            break
    print(dicom_file[list(dicom_file.keys())[i]])# 输出(0028, 0030)标签的值
    print("----------")


# there may be some unsorted image names...
for name in train_case_dict.keys():
    print(name, [train_case_dict[name][i].split("/")[-1] for i in range(7)], end="\n")


for name in test_case_dict.keys():
    print(name, [test_case_dict[name][i].split("/")[-1] for i in range(7)], end="\n")



def resort_names(case_dict, chosen_name, indicator):
    new_path_idx_list = []
    for path in case_dict[chosen_name]:
        name = path.split("/")[-1]
        new_path_idx_list.append(int(name.split(indicator)[1]))
    new_path_idx_list = np.array(new_path_idx_list)
    locs = np.argsort(new_path_idx_list)

    new_path_list = []
    for i in locs:
        new_path_list.append(case_dict[chosen_name][i])

    case_dict[chosen_name]=new_path_list


resort_names(test_case_dict, "CASE37", "I")
resort_names(test_case_dict, "CASE36", "I")
resort_names(test_case_dict, "CASE38", "I")
resort_names(train_case_dict, "CASE16", "I")
resort_names(train_case_dict, "CASE17", "I")
resort_names(train_case_dict, "CASE18", "I")


for name in train_case_dict.keys():
    print(name, [train_case_dict[name][i].split("/")[-1] for i in range(7)], end="\n")


for name in test_case_dict.keys():
    print(name, [test_case_dict[name][i].split("/")[-1] for i in range(7)], end="\n")



output_file_path = '/mnt/e/00-Data/Airway/EXACT09/TEMP/' #"TEMP" #"/data/Airway/EXACT09_3D"

if not os.path.exists(output_file_path):
    os.mkdir(output_file_path)

if not os.path.exists(output_file_path+"/train"):
    os.mkdir(output_file_path+"/train")

if not os.path.exists(output_file_path+"/test"):
    os.mkdir(output_file_path+"/test")

if not os.path.exists(output_file_path+"/train_label"):
    os.mkdir(output_file_path+"/train_label")

def loadFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape
    return img_array, frame_num, width, height

def get_3d_img_for_one_case(img_path_list):
    '''
    input: img_path_list: the list of image paths of one case
    get the 3D image of one case
    '''
    img_3d=[]
    for idx, img_path in enumerate(img_path_list):
        print("progress: "+str(idx/len(img_path_list))+"; "+str(img_path), end="\r")
        img_slice, frame_num, _, _ = loadFile(img_path)
        assert frame_num==1
        img_3d.append(img_slice)
    img_3d=np.array(img_3d)
    return img_3d.reshape(img_3d.shape[0], img_3d.shape[2], img_3d.shape[3])


# preprocess and save one case
img_3d = get_3d_img_for_one_case(train_case_dict["CASE20"])
sitk.WriteImage(sitk.GetImageFromArray(img_3d), output_file_path+"/train/CASE20.nii.gz")


# preprocess and save cases
for case in train_case_dict.keys():
    print(case, end="\n")
    img_3d = get_3d_img_for_one_case(train_case_dict[case])
    sitk.WriteImage(sitk.GetImageFromArray(img_3d),
                    output_file_path+"/train/"+case+'.nii.gz')

for case in test_case_dict.keys():
    print(case, end="\n")
    img_3d = get_3d_img_for_one_case(test_case_dict[case])
    sitk.WriteImage(sitk.GetImageFromArray(img_3d),
                    output_file_path+"/test/"+case+'.nii.gz')


def reverse_img_3d_np(img_3d):
    """
    reverse the 3D image along the 1st axis
    把3D图像沿着第一个轴翻转
    翻转后的图像的第一个channel是原图像的最后一个channel
    """
    img_3d_shape=img_3d.shape
    reverse_img_3d = np.zeros(img_3d_shape)
    # reverse from the 1st channel
    for i in range(img_3d_shape[0]):
        print("progress: "+str(i/img_3d_shape[0]), end="\r")
        reverse_img_3d[img_3d_shape[0]-i-1,:,:]=img_3d[i,:,:]
    return reverse_img_3d


# get labels of the training data
label_file_path = '/mnt/e/00-Data/Airway/EXACT09train_annotation/airways_EXACTtrain' #"/data/Airway/EXACT09_annotation"

for case in train_case_dict.keys():
    print(case)
    img_label_arr, _, _, _ = loadFile(label_file_path+'/'+case+'.mhd')
    sitk.WriteImage(sitk.GetImageFromArray(img_label_arr),
                    output_file_path+"/train_label/"+case+'_label.nii.gz')



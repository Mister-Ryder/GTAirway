#!/usr/bin/env python
# coding: utf-8

# ### Get dataset_info

import os
import numpy as np
import pickle
import edt #cython package
import tqdm


#正确的引用是

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# ### Get pre-cropped EXACT09 dataset info

Precrop_dataset_for_train_path = "./Precrop_dataset_for_EXACT09"
Precrop_dataset_for_train_raw_path = Precrop_dataset_for_train_path+"/image"
Precrop_dataset_for_train_label_path = Precrop_dataset_for_train_path+"/label"

raw_case_name_list = os.listdir(Precrop_dataset_for_train_raw_path)
label_case_name_list = os.listdir(Precrop_dataset_for_train_label_path)

assert raw_case_name_list == label_case_name_list

data_dict = dict()

#for idx, name in enumerate(raw_case_name_list):
for idx, name in tqdm.tqdm(enumerate(raw_case_name_list), total=len(raw_case_name_list)):
    print(f"\rProcessing {name} ({idx + 1}/{len(raw_case_name_list)})", end="")
    #print("process "+str(name)+" | "+str(idx/len(raw_case_name_list)), end="\r")
    data_dict[name.split(".")[0]]={}
    data_dict[name.split(".")[0]]["image"]=Precrop_dataset_for_train_raw_path+"/"+name
    data_dict[name.split(".")[0]]["label"]=Precrop_dataset_for_train_label_path+"/"+name
    label_temp = np.load(Precrop_dataset_for_train_label_path+"/"+name)
    data_dict[name.split(".")[0]]["airway_pixel_num"]=np.sum(label_temp)
    
    label_temp=np.array(label_temp, dtype=np.uint32, order='F')
    label_temp_edt=edt.edt(
        label_temp,
        black_border=True, order='F',
        parallel=1)
    
    data_dict[name.split(".")[0]]["airway_pixel_num_boundary"] = len(np.where(label_temp_edt==1)[0])
    data_dict[name.split(".")[0]]["airway_pixel_num_inner"] = len(np.where(label_temp_edt>1)[0])

save_obj(data_dict, "dataset_info_EXACT09_crops_128")

# ### Get the case names of LIDC-IDRI and EXACT09

EXACT_img_niigz_path = "/mnt/e/00-Data/Airway/EXACT09/TEMP/train"
EXACT_label_niigz_path = "/mnt/e/00-Data/Airway/EXACT09/TEMP/train_label"

EXACT_names = os.listdir(EXACT_img_niigz_path)
EXACT_names.sort()
EXACT_label_names = os.listdir(EXACT_label_niigz_path)
EXACT_label_names.sort()
print(EXACT_names, EXACT_label_names)

EXACT09_names = []
for EXACT09_name in EXACT_names:
    EXACT09_names.append("EXACT09_"+EXACT09_name.split(".")[0])
EXACT09_names = np.array(EXACT09_names)
EXACT09_names = np.unique(EXACT09_names)


# ### Split train/test set

#names = np.concatenate((EXACT09_names, LIDC_IDRI_names))# concatenate the names of EXACT09 and LIDC-IDRI 把EXACT09和LIDC-IDRI的名字连接起来
# you can split train/test by yourself
# just show an example
names = EXACT09_names
# test_names = ['LIDC_IDRI_0698', 'LIDC_IDRI_0710', 'LIDC_IDRI_0810',
#         'LIDC_IDRI_0376', 'EXACT09_CASE13', 'LIDC_IDRI_1004',
#         'EXACT09_CASE08', 'EXACT09_CASE01', 'EXACT09_CASE05',
#         'LIDC_IDRI_0744']
test_names = ['EXACT09_CASE13', 'LIDC_IDRI_1004',
        'EXACT09_CASE08', 'EXACT09_CASE01', 'EXACT09_CASE05']

print("test name: "+str(test_names))
train_names = []
for name in names:
    if name not in test_names:
        train_names.append(name)
train_names=np.array(train_names)
print("train names: "+str(train_names))


# ### Get dataset dict for training

data_dict_EXACT09_128=load_obj("dataset_info/dataset_info_EXACT09_crops_128")

train_test_set_dict_EXACT09_LIDC_IDRI_128={}
train_test_set_dict_EXACT09_LIDC_IDRI_128["train"]={}
train_test_set_dict_EXACT09_LIDC_IDRI_128["test"]={}

for case in data_dict_EXACT09_128.keys():
    if (case.split("_")[0]+"_"+case.split("_")[1]) in train_names:
        train_test_set_dict_EXACT09_LIDC_IDRI_128["train"][case] = data_dict_EXACT09_128[case]
    elif (case.split("_")[0]+"_"+case.split("_")[1]) in test_names:
        train_test_set_dict_EXACT09_LIDC_IDRI_128["test"][case] = data_dict_EXACT09_128[case]


train_test_set_dict_EXACT09_LIDC_IDRI_128["train_names"]=train_names
train_test_set_dict_EXACT09_LIDC_IDRI_128["test_names"]=test_names


#train_test_set_dict_EXACT09_LIDC_IDRI_128["train"]
# it has all info of images crops for training


# ### Get dataset_info for iterative training strategy. First, train with higher frequency on airways of high generations. Next, train with higher frequency on airways of low generations. And repeat the process.

data_dict_org=train_test_set_dict_EXACT09_LIDC_IDRI_128['train']

import copy
data_dict_extended = copy.deepcopy(data_dict_org)

is_more_big = True # higher freq on airways of low gen (thicker airways)
copy_times_I = 10

#我想把enmrated改为tqdm，显示进度条
#for idx, case in enumerate(data_dict_org.keys()):
for idx, case in tqdm.tqdm(enumerate(data_dict_org.keys()), total=len(data_dict_org.keys())):
    print(f"\rProcessing {name} ({idx + 1}/{len(data_dict_org.keys())})", end="")
    if data_dict_org[case]["airway_pixel_num"]>0:
        if is_more_big:
            copy_times_II = np.ceil(data_dict_org[case]["airway_pixel_num_inner"]/data_dict_org[case]["airway_pixel_num_boundary"])
        else:
            if data_dict_org[case]["airway_pixel_num_inner"]==0:
                copy_times_II = np.ceil(data_dict_org[case]["airway_pixel_num_boundary"])
            else:
                copy_times_II = np.ceil(data_dict_org[case]["airway_pixel_num_boundary"]/data_dict_org[case]["airway_pixel_num_inner"])

        for i in range(int(copy_times_I*copy_times_II)):
            data_dict_extended[case+"_copy_"+str(i+1)]=data_dict_org[case]
if is_more_big:
    save_obj(data_dict_extended, "dataset_info/train_dataset_info_EXACT09_LIDC_IDRI_crops_128_extended_"+"more_low_gen_"+str(copy_times_I))
else:
    save_obj(data_dict_extended, "dataset_info/train_dataset_info_EXACT09_LIDC_IDRI_crops_128_extended_"+"more_high_gen_"+str(copy_times_I))
    print("saved")

"""

# ### Get pre-cropped LIDC-IDRI dataset info

Precrop_dataset_for_train_path = "/data/Airway/Precrop_dataset_for_LIDC-IDRI"
Precrop_dataset_for_train_raw_path = Precrop_dataset_for_train_path+"/image"
Precrop_dataset_for_train_label_path = Precrop_dataset_for_train_path+"/label"

raw_case_name_list = os.listdir(Precrop_dataset_for_train_raw_path)
label_case_name_list = os.listdir(Precrop_dataset_for_train_label_path)

assert raw_case_name_list == label_case_name_list

data_dict = dict()

for idx, name in enumerate(raw_case_name_list):
    print("process "+str(name)+" | "+str(idx/len(raw_case_name_list)), end="\r")
    data_dict[name.split(".")[0]]={}
    data_dict[name.split(".")[0]]["image"]=Precrop_dataset_for_train_raw_path+"/"+name
    data_dict[name.split(".")[0]]["label"]=Precrop_dataset_for_train_label_path+"/"+name
    label_temp = np.load(Precrop_dataset_for_train_label_path+"/"+name)
    data_dict[name.split(".")[0]]["airway_pixel_num"]=np.sum(label_temp)
    
    label_temp=np.array(label_temp, dtype=np.uint32, order='F')
    label_temp_edt=edt.edt(
        label_temp,
        black_border=True, order='F',
        parallel=1)
    
    data_dict[name.split(".")[0]]["airway_pixel_num_boundary"] = len(np.where(label_temp_edt==1)[0])
    data_dict[name.split(".")[0]]["airway_pixel_num_inner"] = len(np.where(label_temp_edt>1)[0])

save_obj(data_dict, "dataset_info_LIDC_IDRI_crops_128")

# ### Get the case names of LIDC-IDRI and EXACT09

LIDC_IDRI_img_niigz_path = "/data/Airway/LIDC-IDRI_3D/annotated_data/image"
LIDC_IDRI_label_niigz_path = "/data/Airway/LIDC-IDRI_3D/annotated_data/label"

LIDC_names = os.listdir(LIDC_IDRI_img_niigz_path)
LIDC_names.sort()
LIDC_IDRI_label_names = os.listdir(LIDC_IDRI_label_niigz_path)
LIDC_IDRI_label_names.sort()
print(LIDC_names, LIDC_IDRI_label_names)

LIDC_IDRI_names = []
for LIDC_IDRI_name in LIDC_names:
    LIDC_IDRI_names.append("LIDC_IDRI_"+LIDC_IDRI_name.split(".")[0])
LIDC_IDRI_names = np.array(LIDC_IDRI_names)
LIDC_IDRI_names = np.unique(LIDC_IDRI_names)


data_dict_LIDC_IDRI_128=load_obj("dataset_info/dataset_info_LIDC_IDRI_crops_128")

train_test_set_dict_EXACT09_LIDC_IDRI_128={}
train_test_set_dict_EXACT09_LIDC_IDRI_128["train"]={}
train_test_set_dict_EXACT09_LIDC_IDRI_128["test"]={}

for case in data_dict_LIDC_IDRI_128.keys():
    if (case.split("_")[0]+"_"+case.split("_")[1]+"_"+case.split("_")[2]) in train_names:
        train_test_set_dict_EXACT09_LIDC_IDRI_128["train"][case] = data_dict_LIDC_IDRI_128[case]
    elif (case.split("_")[0]+"_"+case.split("_")[1]+"_"+case.split("_")[2]) in test_names:
        train_test_set_dict_EXACT09_LIDC_IDRI_128["test"][case] = data_dict_LIDC_IDRI_128[case]



train_test_set_dict_EXACT09_LIDC_IDRI_128["train_names"]=train_names
train_test_set_dict_EXACT09_LIDC_IDRI_128["test_names"]=test_names


#train_test_set_dict_EXACT09_LIDC_IDRI_128["train"]
# it has all info of images crops for training


# ### Get dataset_info for iterative training strategy. First, train with higher frequency on airways of high generations. Next, train with higher frequency on airways of low generations. And repeat the process.

data_dict_org=train_test_set_dict_EXACT09_LIDC_IDRI_128['train']

import copy
data_dict_extended = copy.deepcopy(data_dict_org)

is_more_big = True # higher freq on airways of low gen (thicker airways)
copy_times_I = 10

for idx, case in enumerate(data_dict_org.keys()):
    if data_dict_org[case]["airway_pixel_num"]>0:
        if is_more_big:
            copy_times_II = np.ceil(data_dict_org[case]["airway_pixel_num_inner"]/data_dict_org[case]["airway_pixel_num_boundary"])
        else:
            if data_dict_org[case]["airway_pixel_num_inner"]==0:
                copy_times_II = np.ceil(data_dict_org[case]["airway_pixel_num_boundary"])
            else:
                copy_times_II = np.ceil(data_dict_org[case]["airway_pixel_num_boundary"]/data_dict_org[case]["airway_pixel_num_inner"])

        for i in range(int(copy_times_I*copy_times_II)):
            data_dict_extended[case+"_copy_"+str(i+1)]=data_dict_org[case]
if is_more_big:
    save_obj(data_dict_extended, "dataset_info/train_dataset_info_EXACT09_LIDC_IDRI_crops_128_extended_"+"more_low_gen_"+str(copy_times_I))
else:
    save_obj(data_dict_extended, "dataset_info/train_dataset_info_EXACT09_LIDC_IDRI_crops_128_extended_"+"more_high_gen_"+str(copy_times_I))

"""
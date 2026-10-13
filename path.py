# -*- coding: utf-8 -*-
# @Time    : 2025/3/10 9:57
# @Author  : JIA
# @FileName: path.py
# @Software: PyCharm
# @Blog    ：
'''
AeroPath
/home/ubuntu/shared-data/AeroPath/AeroPath/train/image/10_CT_HR.nii.gz
/home/ubuntu/shared-data/AeroPath/AeroPath/train/label/10_CT_HR_label_airways.nii.gz

AIIB23
/home/ubuntu/shared-data/AIIB23/AIIB23_Train_T1/image/AIIB23_100.nii.gz
/home/ubuntu/shared-data/AIIB23/AIIB23_Train_T1/label/AIIB23_100.nii.gz

ATM22
/home/ubuntu/shared-data/ATM22/imagesTr/ATM_001_0000.nii.gz
/home/ubuntu/shared-data/ATM22/labelsTr/ATM_001_0000.nii.gz
/home/ubuntu/shared-data/ATM22/labels_skeletonTr/ATM_001_0000.nii.gz

BRONATLAS-train
/home/ubuntu/shared-data/BRONATLAS/train/image/CASE01-image.nii.gz
/home/ubuntu/shared-data/BRONATLAS/train/label/CASE01-label.nii.gz
BRONATLAS-test
/home/ubuntu/shared-data/BRONATLAS/test/image/CASE15-image.nii.gz
/home/ubuntu/shared-data/BRONATLAS/test/label/CASE15-label.nii.gz

'''
def dice_coefficient_score_calculation(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    intersection = np.sum(pred * label)
    dice_coefficient_score = round(((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(label) + smooth)) * 100, 2)
    return dice_coefficient_score
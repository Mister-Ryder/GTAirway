import SimpleITK as sitk
import os
import tqdm

raw_label_path = "/home/ubuntu/shared-data/LIDC-IDRI_3D/annotated_data/label/0066.nii.gz"
raw_label = sitk.ReadImage(raw_label_path)
direction=raw_label.GetDirection()
print(direction)

# raw_file_path='/home/ubuntu/shared-data/mainfest/LIDC-IDRI_TEMP'
raw_file_path='/home/ubuntu/shared-data/LIDC-IDRI_3D/annotated_data/image'

save_file_path='/home/ubuntu/shared-data/LIDC-IDRI_3D/annotated_data/image2'

# get the list of the raw label
case_dict = dict()
case_names = os.listdir(raw_file_path)
case_names.sort()
for case_name in tqdm.tqdm(case_names):
    case_dict[case_name] = raw_file_path + "/" + case_name
    raw_img_sitk = sitk.ReadImage(case_dict[case_name])
    raw_img_sitk = sitk.Flip(raw_img_sitk, [False, False, True])
    raw_img_sitk.SetDirection(direction)
    sitk.WriteImage(raw_img_sitk, save_file_path + "/" + case_name)
    print(case_dict[case_name])
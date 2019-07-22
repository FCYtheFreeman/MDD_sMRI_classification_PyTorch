import os
import numpy as np
import nibabel as nib
from utils import OsJoin

path = 'E:\DepressionData\sMRIdata\GM_norm_smooth_healthy'
bad_path1 = 'E:\DepressionData\sMRIdata\RAW_MDD\s20090526_09_ZangYF_LSF_ShuiCaiKong-0003-00001-000128-01.nii'
bad_path2 = 'E:\DepressionData\sMRIdata\RAW_MDD\s20090904_03_ZangYF_LSF_LiuZhuo-0003-00001-000128-01.nii'
save_path = 'E:\DepressionData\sMRIdata/s20090526_09_ZangYF_LSF_ShuiCaiKong-0003-00001-000128-01.nii'

def check_size(path):
    '''
    since there're some bad data samples that have different shape from other
    this func can find them out
    '''
    for nii in os.listdir(path):
        if nii[-3:] == 'nii':   # 只读取nii文件
          img = nib.load(OsJoin(path,nii))
        img_data = img.get_data()
        img_arr = np.array(img_data)
        shape1, shape2, shape3 = img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]
        if not (shape1==121 and shape2==145 and shape3==121):
            print(nii)
    #img_arr_cleaned = np.nan_to_num(img_arr)

def resize_bad_data(path, save_path):
    img = nib.load(path)
    affine = img.get_sform()
    img_data = img.get_fdata()
    img_arr = np.array(img_data)
    shape1, shape2, shape3 = img_arr.shape[0], img_arr.shape[1], img_arr.shape[2]
    #zero_slice = np.zeros((shape1, shape2, 1))
    #img_arr = np.append(img_arr, zero_slice)
    img_arr = img_arr.resize((256,256,128))
    img_ = nib.Nifti1Image(img_arr, affine)
    nib.save(img_, save_path)

check_size(path)
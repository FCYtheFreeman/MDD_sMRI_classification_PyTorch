import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import csv
from opts import parse_opts
from utils import OsJoin

opt = parse_opts()
data_type = opt.data_type
csv_dir = OsJoin(opt.data_root_path, 'csv', data_type)

def default_loader(path):
    img_pil = nib.load(path)
    img_arr = np.array(img_pil.get_data())
    img_arr_cleaned = np.nan_to_num(img_arr)    # Replace NaN with zero and infinity with large finite numbers.
    # if path.split('/')[-1] == 's20090904_03_ZangYF_LSF_LiuZhuo-0003-00001-000128-01.nii' or 's20090526_09_ZangYF_LSF_ShuiCaiKong-0003-00001-000128-01.nii':
    #     img_arr_cleaned.resize((256,256,128))   # resize bad samples
    img_pil = torch.from_numpy(img_arr_cleaned)
    return img_pil


class TrainSet(Dataset):

    def __init__(self, fold_id, loader = default_loader):
        with open(csv_dir + '/train_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_train = [OsJoin(opt.data_root_path, row[0]) for row in reader]
        with open(csv_dir + '/train_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_train = [row[1] for row in reader]
        self.image = file_train
        self.label = label_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.image)

class ValidSet(Dataset):
    def __init__(self, fold_id, loader = default_loader):
        with open(csv_dir + '/val_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_valid = [OsJoin(opt.data_root_path, row[0]) for row in reader]
        with open(csv_dir + '/val_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_valid = [row[1] for row in reader]
        self.image = file_valid
        self.label = label_valid
        self.loader = loader

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.image)

class TestSet(Dataset):

    def __init__(self,loader = default_loader):
        with open(csv_dir + '/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_test = [OsJoin(opt.data_root_path, row[0]) for row in reader]
        with open(csv_dir + '/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_test = [row[1] for row in reader]
        self.image = file_test
        self.label = label_test
        self.loader = loader

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.image)

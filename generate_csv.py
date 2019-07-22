import csv
import numpy as np
import os
from math import ceil
from utils import OsJoin
from sklearn.model_selection import KFold
from opts import parse_opts

opt = parse_opts()
root = opt.data_root_path
health_dir = opt.data_type + '_healthy'
MDD_dir = opt.data_type + '_MDD'
csv_save_dir = OsJoin('csv/', opt.data_type)
test_ratio = 0.1
n_fold = 5

data_health = []
label_health = []
data_MDD = []
label_MDD = []
for filename in os.listdir(OsJoin(root, health_dir)):
    data_health.append(OsJoin(health_dir, filename))
    label_health.append(0) # 0 for health label
for filename in os.listdir(OsJoin(root, MDD_dir)):
    data_MDD.append(OsJoin(MDD_dir, filename))
    label_MDD.append(1) # 1 for MDD label

health_list = np.array([data_health, label_health]).transpose()
MDD_list = np.array([data_MDD, label_MDD]).transpose()  # 都是按照名称顺序排列读入


np.random.seed(opt.manual_seed)  #固定seed后每种数据都按照同一shuffle顺序排列
np.random.shuffle(health_list)  # 打乱行顺序
np.random.shuffle(MDD_list)
#health_list_rand = np.random.permutation(health_list)
# rand_rows = np.arange(health_list.shape[0])
# health_list = health_list[rand_rows]


# # down sampling
# MDD_list = MDD_list[0:health_list.shape[0]]

n_test_health = ceil(health_list.shape[0] * test_ratio)  # number of test samples
n_test_MDD = ceil(MDD_list.shape[0] * test_ratio)  # number of test samples
n_train_val_health = health_list.shape[0] - n_test_health  # number of trainning samples
n_train_val_MDD = MDD_list.shape[0] - n_test_MDD  # number of trainning samples

train_val_list_health = health_list[0:n_train_val_health, :]
train_val_list_MDD = MDD_list[0:n_train_val_MDD, :]
test_list_health = health_list[n_train_val_health:health_list.shape[0], :]
test_list_MDD = MDD_list[n_train_val_MDD:MDD_list.shape[0], :]

kf = KFold(n_splits=opt.n_fold, shuffle=False)
n = 0
names = locals()
for train_index, val_index in kf.split(train_val_list_health):
    n += 1
    names['train_fold%s_health'%n] = train_val_list_health[train_index]
    names['val_fold%s_health' % n] = train_val_list_health[val_index]
n = 0
for train_index, val_index in kf.split(train_val_list_MDD):
    n += 1
    names['train_fold%s_MDD'%n] = train_val_list_MDD[train_index]
    names['val_fold%s_MDD' % n] = train_val_list_MDD[val_index]

names2 = locals()
for i in range(1, n_fold+1):
    names2['train_list_fold%s'%i] = np.vstack((names2.get('train_fold%s_health'%i), names2.get('train_fold%s_MDD'%i)))
    names2['val_list_fold%s'%i] = np.vstack((names2.get('val_fold%s_health'%i), names2.get('val_fold%s_MDD'%i)))
    np.random.seed(opt.manual_seed)
    np.random.shuffle(names2['train_list_fold%s'%i])
    np.random.shuffle(names2['val_list_fold%s'%i])

test_list = np.vstack((test_list_health, test_list_MDD))    # 按行堆叠
np.random.seed(opt.manual_seed)
np.random.shuffle(test_list)

csv_save_path = OsJoin(root, csv_save_dir)
if not os.path.exists(csv_save_path):
    os.makedirs(csv_save_path)

for i in range(1, n_fold+1):
    with open(OsJoin(csv_save_path, 'train_fold%s.csv'%i), 'w', newline='') as f:  # 设置文件对象
        f_csv = csv.writer(f)
        f_csv.writerows(names2.get('train_list_fold%s'%i))
    with open(OsJoin(csv_save_path, 'val_fold%s.csv'%i), 'w', newline='') as f:  # 设置文件对象
        f_csv = csv.writer(f)
        f_csv.writerows(names2.get('val_list_fold%s'%i))


with open(OsJoin(csv_save_path, 'test.csv'), 'w', newline='') as f:  # 设置文件对象
    f_csv = csv.writer(f)
    f_csv.writerows(test_list)
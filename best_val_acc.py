import os
from utils import  OsJoin

'''
this script is for calculating best average val accuracy for n fold cross validation and finding which epoch it's in.
'''

log_dir = 'E:\GitHub_repos\MRI_pytorch/results\GM_norm_smooth/resnet10'

n_fold_lines = []
for i in range(1, 6): # n folds
    log_path = OsJoin(log_dir, 'logs_fold%s'%str(i), 'val.log')
    with open(log_path, 'r') as file:
        lines = file.readlines()
        # delete blank row
        clean_lines = []
        for line in lines:
            if line != '\n':
                clean_lines.append(line.strip('\n'))
        n_fold_lines.append(clean_lines)
    file.close()

best_epoch, best_acc = 1, 0. # initialize
for fold1_line,fold2_line,fold3_line,fold4_line,fold5_line in zip(n_fold_lines[0][1:],n_fold_lines[1][1:],n_fold_lines[2][1:],n_fold_lines[3][1:],n_fold_lines[4][1:]): #excude header
    words1 = fold1_line.split('\t') # split by tabs
    words2 = fold2_line.split('\t')
    words3 = fold3_line.split('\t')
    words4 = fold4_line.split('\t')
    words5 = fold5_line.split('\t')
    avg_acc = (float(words1[2]) + float(words2[2]) + float(words3[2]) + float(words4[2]) + float(words5[2]))/5.
    if avg_acc >= best_acc:
        best_epoch = words1[0]
        best_acc = avg_acc
best_accs = [best_epoch, best_acc]
print(best_accs)
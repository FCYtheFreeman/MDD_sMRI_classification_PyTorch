import csv
import os

class AverageMeter(object):
    '''computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count =0

    def update(self,val,n=1):
        self.val = val
        self.sum += val*n
        self.count +=n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t') #\t为一个tab

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, labels):
    batch_size = labels.size(0)

    _, pred = outputs.topk(k=1, dim=1, largest=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1))
    n_correct_elems = correct.float().sum().data

    return n_correct_elems / batch_size

def calculate_recall(outputs, labels):
    _, pred = outputs.topk(k=1, dim=1, largest=True)
    pred = pred.t() # 转置成行
    TP = ((pred.data == 1) & (labels.data == 1)).cpu().float().sum().data
    #TN = ((pred.data == 0) & (labels.data == 0)).cpu().float().sum().data
    FN = ((pred.data == 0) & (labels.data == 1)).cpu().float().sum().data
    #FP = ((pred.data == 1) & (labels.data == 0)).cpu().float().sum().data
    #p = TP / (TP + FP)  #precision
    r = TP / (TP + FN)  #recall
    #F1 = 2 * r * p / (r + p)

    return r

# def calculate_best_metric(epoch_metric):
#     best_metric = max(epoch_metric[:,1])
#     return

def OsJoin(*args):
    p = os.path.join(*args)
    p = p.replace('\\', '/')
    return p

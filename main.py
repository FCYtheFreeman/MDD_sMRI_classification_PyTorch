import os
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from dataset import TrainSet, ValidSet
from utils import Logger, OsJoin
from train import train_epoch
from validation import val_epoch
from tensorboardX import SummaryWriter

def run(fold_id, opt):
    if opt.root_path != '':
        opt.result_path = OsJoin(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        if opt.resume_path:
            opt.resume_path = OsJoin(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = OsJoin(opt.root_path, opt.pretrain_path)
    opt.arch ='{}-{}'.format(opt.model_name,opt.model_depth)
    #print(opt)

    print('-'*50, 'RUN FOLD %s'%str(fold_id), '-'*50)

    model, parameters = generate_model(opt)
    # print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if not opt.no_train:
        training_data = TrainSet(fold_id=fold_id)
        train_loader = DataLoader(training_data, batch_size = opt.batch_size,shuffle = True,
                                                num_workers = opt.n_threads, pin_memory =True)
        log_path = OsJoin(opt.result_path, opt.data_type, opt.model_name+str(opt.model_depth),
                                                                    'logs_fold%s'%str(fold_id))
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        train_logger = Logger(
            OsJoin(log_path,'train.log'),
            ['epoch','loss','acc','lr'])

        train_batch_logger = Logger(
            OsJoin(log_path, 'train_batch.log'),
            ['epoch','batch','iter','loss','acc','lr'])

        optimizer = optim.Adam(parameters, lr=opt.learning_rate, weight_decay=opt.weight_decay)
        #optimizer = optim.SGD(parameters, lr=opt.learning_rate, momentum=0.9)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                   factor=opt.lr_decay_factor, patience =opt.lr_patience)
    if not opt.no_val:
        validation_data = ValidSet(fold_id=fold_id)
        val_loader = DataLoader(validation_data, batch_size = opt.batch_size, shuffle = False,
                                                    num_workers = opt.n_threads, pin_memory=True)
        val_logger =  Logger(OsJoin(log_path,'val.log'),['epoch','loss','acc','recall'])

    if opt.pretrain_path:
        print('loading checkpoint{}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch==checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    #print('run')
    writer = SummaryWriter(logdir='events/{}/{}_{}/{}'.format(opt.data_type, opt.model_name,
                                                              opt.model_depth, 'fold%s'%str(fold_id)))
    for i in range(opt.begin_epoch,opt.n_epochs+1):
        if not opt.no_train:
            train_epoch(i, fold_id, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger, writer)
        if not opt.no_val:
            validation_loss = val_epoch(i,val_loader, model, criterion, opt, val_logger, writer)
        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar('lr', lr, i)
    writer.close()
    print('-'*47, 'FOLD %s FINISHED'%str(fold_id), '-'*48)

# if opt.test:
#     test_data = TestSet()
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size = opt.batch_size, shuffle=False,
#                                                         num_workers = opt.n_threads, pin_memory=True)
#     test.test(test_loader, model, opt, test_data.label)

if __name__ == '__main__':
    opt = parse_opts()
    # 交叉验证
    for fold_id in range(1, opt.n_fold + 1):
        run(fold_id, opt)
import torch
from torch.autograd import Variable
import os
from utils import OsJoin
import time
from utils import AverageMeter,calculate_accuracy

def train_epoch(epoch, fold_id, data_loader, model, criterion, optimizer,
                opt, epoch_logger, batch_logger, writer):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses= AverageMeter()
    accuracies = AverageMeter()

    writer = writer
    end_time  = time.time()
    for i ,(inputs,labels) in enumerate(data_loader):
        data_time.update(time.time()-end_time)
        labels = list(map(int, labels))
        inputs = torch.unsqueeze(inputs,1)  #在 1 的位置加一个维度
        inputs = inputs.type(torch.FloatTensor)
        if not opt.no_cuda:
            labels = torch.LongTensor(labels).cuda(async = True)
        inputs = Variable(inputs)
        labels = Variable(labels)
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        acc = calculate_accuracy(outputs,labels)

        losses.update(loss.data,inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time()-end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch':epoch,
            'batch':i+1,
            'iter':(epoch-1)*len(data_loader)+(i-1),
            'loss':round(losses.val.item(), 4),
            'acc':round(accuracies.val.item(), 4),
            'lr':optimizer.param_groups[0]['lr']
        })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch, i + 1, len(data_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': round(losses.avg.item(), 4),
        'acc': round(accuracies.avg.item(), 4),
        'lr': optimizer.param_groups[0]['lr']
    })

    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/accuracy', accuracies.avg, epoch)

    if opt.save_weight:
        if epoch % opt.checkpoint == 0:
            save_dir =OsJoin(opt.result_path, opt.data_type, opt.model_name + str(opt.model_depth),
        'weights_fold%s'%fold_id)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = OsJoin(save_dir,
                        '{}{}_weights_fold{}_epoch{}.pth'.format(opt.model_name, opt.model_depth, fold_id, epoch))
            states = {
                'fold': fold_id,
                'epoch': epoch,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states,save_path)




import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy, calculate_recall

def val_epoch(epoch,data_loader,model,criterion,opt,logger, writer):
    print('validation at epoch {}'.format(epoch) )
    model.eval()

    batch_time =AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    recalls = AverageMeter()

    writer = writer

    end_time = time.time()
    for i ,(inputs,labels) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        labels = list(map(int,labels))
        inputs = torch.unsqueeze(inputs,1)
        inputs = inputs.type(torch.FloatTensor)

        if not opt.no_cuda:
            labels = torch.LongTensor(labels).cuda(async=True)
        with torch.no_grad():
            inputs = Variable(inputs)
            labels = Variable(labels)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            recall = calculate_recall(outputs, labels)

        losses.update(loss.data, inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        recalls.update(recall, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'Recall {recall.val:.3f} ({recall.avg:.3f})'.format(
            epoch,
            i + 1,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            acc=accuracies,
            recall=recalls))

    logger.log({'epoch': epoch, 'loss': round(losses.avg.item(), 4),
                'acc': round(accuracies.avg.item(), 4), 'recall': round(recalls.avg.item(), 4)})
    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/accuracy', accuracies.avg, epoch)
    writer.add_scalar('val/recall', recalls.avg, epoch)

    return losses.avg

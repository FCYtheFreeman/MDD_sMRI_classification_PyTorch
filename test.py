from torch.autograd import Variable
import time
from utils import OsJoin
from utils import AverageMeter,calculate_accuracy,calculate_recall
import torch
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from dataset import TestSet
from utils import Logger
from torch import nn

opt = parse_opts()

def test_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('test at epoch {}'.format(epoch))
    model.eval()

    batch_time =AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    recalls = AverageMeter()

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

    logger.log({'epoch': epoch, 'loss': round(losses.avg.item(), 4), 'acc': round(accuracies.avg.item(), 4),
                                                                'recall': round(recalls.avg.item(), 4)})

if opt.test:
    if opt.resume_path:
        opt.resume_path = OsJoin(opt.root_path, opt.resume_path)
    test_data = TestSet()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = opt.batch_size, shuffle=False,
                                                        num_workers = 0, pin_memory=True)
    model, parameters = generate_model(opt)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
    log_path = OsJoin(opt.result_path, opt.data_type, opt.model_name + str(opt.model_depth))
    test_logger = Logger(
        OsJoin(log_path, 'test.log'), ['epoch', 'loss', 'acc', 'recall'])
    print('loading checkpoint{}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    assert opt.arch == checkpoint['arch']

    model.load_state_dict(checkpoint['state_dict'])

    test_epoch(1, test_loader, model, criterion, opt, test_logger)

# def calculate_test_results(output_buffer,sample_id,test_results,labels):
#     outputs =torch.stack(output_buffer)
#     average_score = torch.mean(outputs,dim=0)
#     sorted_scores,locs = torch.topk(average_score,k=1)
#     results=[]
#     for i in range(sorted_scores.size(0)):
#         score = copy.deepcopy(sorted_scores[i])
#         if isinstance(score, torch.Tensor):
#             score = score.data.cpu().numpy()
#             score = score.item()
#         results.append({
#             'label':labels[i],
#             'score':score
#         })
#     test_results['results'][sample_id] = results
#
# def test(data_loader, model, opt, labels):
#     print('test')
#
#     model.eval()
#
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#
#     end_time = time.time()
#     output_buffer = []
#     sample_id = ''
#     test_results = {'results': {}}
#     with torch.no_grad():
#         for i, (inputs, targets) in enumerate(data_loader):
#             data_time.update(time.time() - end_time)
#
#             inputs = torch.unsqueeze(inputs, 1)  # 在 1 的位置加一个维度
#             inputs = Variable(inputs)
#             outputs = model(inputs)
#             # if not opt.no_softmax_in_test:
#             #outputs = F.softmax(outputs)
#
#             for j in range(outputs.size(0)):
#                 if not (i == 0 and j == 0):
#                     calculate_test_results(output_buffer, sample_id, test_results, labels)
#                     output_buffer = []
#                 output_buffer.append(outputs[j].data.cpu())
#                 sample_id = labels[j]
#             if (i % 100) == 0:
#                 with open(
#                         OsJoin(opt.result_path, '{}.json'.format(
#                             opt.test_subset)), 'w') as f:
#                     json.dump(test_results, f)
#
#             batch_time.update(time.time() - end_time)
#             end_time = time.time()
#
#             print('[{}/{}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
#                       i + 1,
#                       len(data_loader),
#                       batch_time=batch_time,
#                       data_time=data_time))
#     with open(
#             OsJoin(opt.result_path, opt.data_type, opt.model_name, str(opt.model_depth), '{}.json'.format(opt.test_subset)),
#             'w') as f:
#         json.dump(test_results, f)

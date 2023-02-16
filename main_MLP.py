# Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import argparse
import os
import shutil
import time
import numpy as np
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import FPHA_dataset


from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='STGST MLP')
parser.add_argument('--epochs', default=1000, type= int, metavar='N', help='number of total epochs')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--workers',default=2,type=int, metavar='N')
parser.add_argument('--num_classes', default=45, type=int, metavar='N')
parser.add_argument('-b', '--batch_size',default=16, type=int,metavar='N')
parser.add_argument('--lr',default=0.001,type = float, metavar='LR')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=80, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--sum_freq', default=40, type=int, metavar='N', help='print frequency (default: 40)')
parser.add_argument('--save_freq', default=1, type=int, metavar='N', help='save checkpoint frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-sp', '--summary_path', default='', type=str, metavar='PATH', help='path to store summary event file')
parser.add_argument('-cp','--checkpoint_path', default='', type=str, metavar='PATH', help='path to store checkpoint')
parser.add_argument('-op', '--output_path', default='', type=str, metavar='PATH', help='path to store test output')
parser.add_argument('--suffix', default ='', type = str, help = 'suffix of summmary and checkpoint dir')
parser.add_argument('--lr_path', default='', type=str, metavar='PATH', help='path to lr file')

parser.add_argument('--optimizer', type=str)
parser.add_argument('--dataroot', type = str,
                    metavar='PATH',help='path to image data root')
parser.add_argument('--thres', type = float,
                    help = 'threshold for scattering tree pruning')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')


# class MLPs(nn.Module):
#     def __init__(self, class_num=20, midnum = 256, nodeNum=None):
#         super(MLPs, self).__init__()
#         self.nodeNum = nodeNum
#         self.mlp1 = nn.Linear(in_features=self.nodeNum*63, out_features=midnum, bias=True)
#         self.mlp2 = nn.Linear(in_features=midnum, out_features=class_num, bias=True)
#         self.relu = nn.ReLU()
#         # self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.mlp1(x)
#         x = self.relu(x)
#         x = self.mlp2(x)
#         return x

# class MLPs(nn.Module):
#     def __init__(self, class_num=20, midnum = 256, nodeNum=None):
#         super(MLPs, self).__init__()
#         self.nodeNum = nodeNum
#         self.mlp1 = nn.Linear(in_features=self.nodeNum*63, out_features=midnum, bias=True)
#         self.bn1 = nn.BatchNorm1d(num_features=midnum)
#         self.mlp2 = nn.Linear(in_features=midnum, out_features=class_num, bias=True)
#         self.relu = nn.ReLU()
#         # self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.mlp1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.mlp2(x)
#         return x

class MLPs(nn.Module):
    def __init__(self, class_num=20, midnum = 128, nodeNum=None):
        super(MLPs, self).__init__()
        self.nodeNum = nodeNum
        self.mlp1 = nn.Linear(in_features=self.nodeNum*63, out_features=midnum, bias=True)
        self.dropout1 = nn.Dropout(0.5)
        self.mlp2 = nn.Linear(in_features=midnum, out_features=class_num, bias=True)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.mlp2(x)
        return x


best_prec1 = 0

def main():
    global args, best_prec1, writer
    args = parser.parse_args()

    args.summary_path = os.path.join(args.summary_path, args.suffix)
    if not os.path.exists(args.summary_path):
        os.makedirs(args.summary_path)
    
    writer = SummaryWriter(args.summary_path)

    # args.checkpoint_path = os.path.join(args.checkpoint_path,args.suffix)
    # if not os.path.exists(args.checkpoint_path):
    #     os.makedirs(args.checkpoint_path)

    # args.output_path = os.path.join(args.output_path, args.suffix)
    # if not os.path.exists(args.output_path):
    #     os.makedirs(args.output_path)

    # model = inception_v3(pretrained=True)

    train_dataset = FPHA_dataset.FPHADataset(args.dataroot, split='train', thres=args.thres, normalize=True)
    test_dataset = FPHA_dataset.FPHADataset(args.dataroot, split='test', thres=args.thres, normalize=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                   shuffle=True, num_workers=args.workers, pin_memory=True)
    global epoch_len
    epoch_len = len(train_loader)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                  batch_size=args.batch_size, shuffle=False,
                  num_workers=args.workers, pin_memory=True)

    model = MLPs(class_num=args.num_classes, midnum=128, nodeNum=train_dataset.nodeNum).cuda()
    print('Reserve Node Num:', train_dataset.nodeNum)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                    args.lr,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise RuntimeError()

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_path)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = evaluate(test_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     #'arch': args.arch,
        #     'model_state': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer' : optimizer.state_dict(),
        # }, is_best, epoch)


def train(trainloader, model, criterion, optimizer, epoch):

    # batch_time = AverageMeter()
    # data_time = AverageMeter()
    lossM = AverageMeter()
    acc1M = AverageMeter()

    model.train()

    end = time.time()

    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        # data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output.detach(), target)[0]
        lossM.update(loss.item(), input.size(0))
        acc1M.update(acc1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # if i % args.print_freq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
        #         epoch, i, len(trainloader), batch_time=batch_time,
        #         data_time=data_time, loss=losses, top1=top1))

        # global_step = epoch * epoch_len + i
        # if i % args.sum_freq == 0:
        #     writer.add_scalar('train_loss', loss.item(), global_step)
        #     writer.add_scalar('train_prec', prec.item(), global_step)
    # global_step = epoch * epoch_len + epoch_len - 1
    writer.add_scalar('epochavg_train_loss', lossM.avg, epoch)
    writer.add_scalar('epochavg_train_acc1', acc1M.avg, epoch)



def evaluate(testloader, model, criterion, epoch):
    # epoch < 0 means no summary
    # batch_time = AverageMeter()
    lossM = AverageMeter()
    acc1M = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(testloader):
        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output.detach(), target)[0]
        lossM.update(loss.item(), input.size(0))
        acc1M.update(acc1.item(), input.size(0))

        # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'acc1 {acc1M.val:.3f}% ({acc1M.avg:.3f}%)'.format(
        #         i, len(testloader), batch_time=batch_time, loss=lossM,
        #         acc1M=acc1M))
        # if i == 0:
        #     out_array = output.detach().cpu().numpy()
        # else:
        #     out_array = np.concatenate((out_array,output.detach().cpu().numpy()), axis=0)

    print(' * Accuracy {acc1M.avg:.3f}% '.format(acc1M=acc1M))

    if epoch >= 0:
        # global_step = epoch * epoch_len + epoch_len - 1
        writer.add_scalar('test_loss', lossM.avg, epoch)
        writer.add_scalar('test_acc', acc1M.avg, epoch)
        # print('Saving output:')
        # np.save(os.path.join(args.output_path, 'out{:0>3}.npy'.format(epoch)), out_array)

    return acc1M.avg

def save_checkpoint(state, is_best, epoch):
    if is_best:
        filepath = os.path.join(args.checkpoint_path, 'model{:0>3}best.pth'.format(epoch))
        torch.save(state, filepath)
    elif epoch % args.save_freq == 0:
        filepath = os.path.join(args.checkpoint_path, 'model{:0>3}.pth'.format(epoch))
        torch.save(state, filepath)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def adjust_learning_rate(optimizer, epoch, file_path):
    f = open(file_path)
    lines = f.readlines()
    lines = [i.strip() for i in lines]
    lines = [i for i in lines if i]
    f.close()

    for line in lines:
        t, l = line.split()
        t = int(t)
        l = float(l)

        if epoch == t:
            lr_times = l
            break
    else:
        lr_times = 1

    for param_group in optimizer.param_groups:
        tmp_lr = param_group['lr']
        lr = tmp_lr * lr_times
        param_group['lr'] = lr
    # global_step = epoch * epoch_len
    print('epoch' + str(epoch) + ' learning rate times: ' + str(lr_times) + '\n')
    # writer.add_scalar('lr', lr, global_step)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.detach().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.detach().view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

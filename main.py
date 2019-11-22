import argparse
import os
import random
import shutil
import time
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
import numpy as np
import mmgresnet, mmgdensenet
from dataloader import get_loader
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])
                     and ("resnet" in name or "densenet" in name))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset(pickle)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet34)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=96, type=int,
                    metavar='N',
                    help='mini-batch size (default: 96), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('--fold-number', default=5, type=int, metavar='N',
                    help='The fold number in cross validation')
parser.add_argument('--current-fold-number', default=1, type=int, metavar='N',
                    help='Current fold number in cross validation')
parser.add_argument('--use-compressed', action='store_true',
                    help='use compressed image in evaluation time')
parser.add_argument('--prefix-image-dir-path', default='', type=str, metavar='PATH',
                    help='path to directory that contains images (default: none)')
parser.add_argument('--image-size', default=(960, 640), type=int, nargs=2,
                    help='image size to use in deep learning network')

best_auc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    main_worker(args)


def main_worker(args):
    global best_auc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if 'resnet' in args.arch.lower():
        model = mmgresnet.__dict__[args.arch]()
    elif 'densenet' in args.arch.lower():
        model = mmgdensenet.__dict__[args.arch]()
    else:
        raise ValueError('Cannot find proper model from given architecture')

    args.map_size = model.get_map_size(args.image_size)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_auc1 = checkpoint['best_auc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if not os.path.isfile(args.data):
        raise ValueError("no file found at {}".format(args.data))

    train_loader, val_loader = get_loader(args)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        print("====={}th epoch========================== best auc is {}".format(epoch, best_auc1))
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        auc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = auc1 > best_auc1
        best_auc1 = max(auc1, best_auc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_auc1': best_auc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, prefix="{}-{}-{}-".format(
            args.arch, args.fold_number, args.current_fold_number)
        )


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    results = defaultdict(lambda: {'label': None, 'predictions': []})

    with torch.no_grad():
        end = time.time()
        for i, (images, target, case_ids, case_labels) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            predictions = np.max(torch.sigmoid(output).cpu().numpy(), axis=(1, 2))

            for idx, case_id in enumerate(case_ids):
                results[case_id]['label'] = (case_labels[idx] == 2).cpu().numpy()
                results[case_id]['predictions'].append(predictions[idx])

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    labels, predictions = [], []

    for v in results.values():
        labels.append(v['label'])
        predictions.append(np.max(np.array(v['predictions'])))

    labels = np.array(labels).astype(np.float32)
    predictions = np.array(predictions)

    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    for threshold in [0.1, 0.15, 0.2]:
        y_pred = predictions.copy()
        y_pred[predictions > threshold] = 1
        y_pred[predictions <= threshold] = 0

        tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print('threshold : {}'.format(threshold))
        print('         calculated accuracy is {}'.format(accuracy_score(labels, y_pred)))
        print('         calculated specificity is {}'.format(specificity))
        print('         calculated sensitivity is {}'.format(sensitivity))
    print("calculated auc is {}".format(roc_auc))
    return roc_auc


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix+filename)
    if is_best:
        shutil.copyfile(prefix+filename, prefix+'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every (total epochs / 3) epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs//3)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()

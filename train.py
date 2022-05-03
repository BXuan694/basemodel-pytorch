from data.config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import numpy as np
import time
cudnn.benchmark = True

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

# 设置新学习率
def set_lr(optimizer_, newLr_):
    for paramGroup_ in optimizer_.param_groups:
        paramGroup_['lr'] = newLr_

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    #print(output)
    #print(pred)


    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))
    #print(correct)

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# 计算warmup学习率
def get_warmup_lr(curIter_, totalIters_, baseLr_, warmupRatio_, warmUpOption='linear'):
    if warmUpOption == 'constant':
        warmupLr = baseLr_ * warmupRatio_ 
    elif warmUpOption == 'linear':
        k = (1 - curIter_ / totalIters_) * (1 - warmupRatio_)
        warmupLr = baseLr_ * (1 - k)
    elif warmUpOption == 'exp':
        k = warmupRatio_**(1 - curIter_ / totalIters_)
        warmupLr = baseLr_ * k
    return warmupLr


def train(globalStartEpoch, totalEpoches):
    inputSize=64
    transform_train = transforms.Compose([
        transforms.RandomCrop(inputSize, padding=50),
        #transforms.RandomResizedCrop(inputSize, scale=(1, 1), ratio=(1, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_valid = transforms.Compose([
        transforms.Resize([inputSize, inputSize]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    if cfg.dataset.structure=="default":
        trainset = datasets.ImageFolder(cfg.dataset.train_info, transform=transform_train)
        validset = datasets.ImageFolder(cfg.dataset.valid_info, transform=transform_valid)
    else:
        pass
    batchSize = cfg.imgs_per_gpu * cfg.num_gpus

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, pin_memory=True, num_workers=cfg.workers_per_gpu)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batchSize, shuffle=False, pin_memory=True, num_workers=cfg.workers_per_gpu)
    
    model = cfg.backbone
    model.cuda()
    model.train()
    
    lrOri = cfg.optimizer['lr']
    lrStages = cfg.lr_config["step"]
    lrList = np.full(totalEpoches, lrOri)
    for ii in range(len(lrStages)):
        lrList[lrStages[ii]:]*=0.1
    print("starting epoch: ", globalStartEpoch)
    print("lr adapting stages: ", end=' ')
    for ii in range(len(lrStages)):
        print(cfg.lr_config["step"][ii], end=" ")
    print("\ntotal training epoches: ", totalEpoches)

    optimizer_config = cfg.optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=optimizer_config['lr'], momentum=optimizer_config['momentum'], weight_decay=optimizer_config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    batchSize = cfg.imgs_per_gpu * cfg.num_gpus
    epochSize = len(trainset) // batchSize  
    # nums of trained epoches, idx of epoch to start
    pastEpoches = globalStartEpoch
    # nums of trained iters, idx of iter to start
    pastIters = (globalStartEpoch-1) * epochSize
    # nums of left epoches
    leftEpoches = totalEpoches - pastEpoches + 1
    # nums of left iters
    leftIters = leftEpoches * epochSize

    print('##### begin train ######')
    top1 = AverageMeter()
    top5 = AverageMeter()
    currentIter = 0

    for epoch in range(leftEpoches):
        currentEpoch = epoch + pastEpoches

        # 终止训练
        if currentEpoch >= totalEpoches:
            print("Current epoch is larger than setting epoch nums, training stop.")
            return

        # 仅用于打印
        for batchIdx, (inputs, targets) in enumerate(trainloader):
            iterStartTime = time.time()

            if cfg.lr_config['warmup'] is not None and pastIters < cfg.lr_config['warmup_iters']:
                cur_lr = get_warmup_lr(pastIters, cfg.lr_config['warmup_iters'],
                                        optimizer_config['lr'], cfg.lr_config['warmup_ratio'],
                                        cfg.lr_config['warmup'])
            else:
                cur_lr = lrList[currentEpoch]
            set_lr(optimizer, cur_lr)

            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            loss.backward()
            optimizer.step()

            leftIters -= 1
            pastIters += 1
            currentIter += 1

            showIters = 10
            if batchIdx%int(showIters) == 0 and batchIdx != 0:
                iterLastTime = time.time() - iterStartTime
                left_seconds = iterLastTime * leftIters
                left_minutes = left_seconds / 60.0
                left_hours = left_minutes / 60.0
                left_days = left_hours // 24
                left_hours = left_hours % 24

                out_srt = 'Epoch:[' + str(currentEpoch) + ']/[' + str(totalEpoches) + '],'
                out_srt += '[' + str(batchIdx) + ']/' + str(len(trainloader)) + '], '
                out_srt += 'left_time: ' + str(left_days)+'days '+format(left_hours,'.2f')+'hours. '
                print(out_srt+'Loss {loss:.5f}\t'
                    'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                    'Prec@5 {top5.val:.2f} ({top5.avg:.2f})\t[lr:{lr:.6g}]'.format(
                    currentEpoch, totalEpoches, batchIdx, len(trainloader),
                    loss=loss.item(), top1=top1, top5=top5, lr=cur_lr))

        leftEpoches -= 1
        top1.reset()
        top5.reset()
        save_name = "./weights/" + cfg.name + "_epoch_" + str(currentEpoch) + ".pth"
        torch.save(model.state_dict(), save_name)

if __name__ == '__main__':
    train(globalStartEpoch=cfg.epoch_iters_start, totalEpoches=cfg.total_epoch)   #设置本次训练的起始epoch
from data.config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import os, sys
import numpy as np
import time
from PIL import Image
import math
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
        break

# 评估精度
def accuracy(output, target, topk=(1,)):
    """
    计算topk准确率.
    output:
     list，shape和topk一致, 元素是输入batch中对应topk的准确率。
    """
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
    """
    预热期初始的学习率：baseLr_*warmupRatio_，最终学习率：baseLr_，增长方式由warmUpOption决定。
    input：
     curIter_: int，当前iter轮次。
     totalIters_: int，预热期总iter轮次。
     baseLr_: float，基准学习率, 预热期结束时的学习率。
     warmupRatio_: float，学习率预热比率，warmupRatio_*baseLr_表示第0iter的学习率。
     warmUpOption: str，学习率增长形式。
    output：
     float，当前(curIter_)轮次的学习率。
    """

    # 学习率从baseLr_*warmupRatio_开始，增长至baseLr_
    if warmUpOption == 'constant':
        warmupLr = baseLr_*warmupRatio_ 
    elif warmUpOption == 'linear':
        k = (1 - curIter_/totalIters_) * (1-warmupRatio_)
        warmupLr = baseLr_*(1-k)
    elif warmUpOption == 'exp':
        k = warmupRatio_**(1 - curIter_/totalIters_)
        warmupLr = baseLr_*k
    return warmupLr

def default_loader(path):
    return Image.open(path).convert('RGB')

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        with open(txt, 'r') as f:
            for line in f:
                words = line.strip().split()
                imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

def train(globalStartEpoch, totalEpoches):
    #------------------------------------data--------------------------------
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(cfg.inputShape, scale=(0.4, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_valid = transforms.Compose([
        transforms.Resize(cfg.inputShape[0]),
        transforms.CenterCrop(cfg.inputShape),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if cfg.dataset.structure=="default":
        trainset = datasets.ImageFolder(cfg.dataset.trainRoot, transform=transform_train)
        validset = datasets.ImageFolder(cfg.dataset.validRoot, transform=transform_valid)
    elif cfg.dataset.structure=="txt":
        trainset = MyDataset(txt=os.path.join(cfg.dataset.trainRoot, 'dataset-trn.txt'), transform=transform_train)
        validset = MyDataset(txt=os.path.join(cfg.dataset.validRoot, 'dataset-val.txt'), transform=transform_valid)

    validloader = DataLoader(validset, batch_size=cfg.batchSize, shuffle=False, pin_memory=True, num_workers=cfg.workers_per_gpu)
    '''
    for idx, (data, target) in enumerate(validloader):
        if(idx%10==0):
            print(str(idx)+' '+str(target))
    '''
    #------------------------------------model--------------------------------
    model = cfg.backbone
    
    # 抽取与训练权重
    model_dict = model.state_dict()
    model_official = cfg.model_official
    model_official = {k: v for k, v in model_official.state_dict().items() if k in model_dict}
    model_dict.update(model_official)
    model.load_state_dict(model_dict)

    model.cuda()

    #------------------------------------optimizer--------------------------------
    lrOri = cfg.optimizer['lr']
    lrList = np.full(totalEpoches, lrOri)
    if cfg.lr_config["policy"]=='step':
        lrStages = cfg.lr_config["step"]
        for ii in range(len(lrStages)):
            lrList[lrStages[ii]:]*=0.1
        print("lr adapting stages: ", end=' ')
        for ii in range(len(lrStages)):
            print(cfg.lr_config["step"][ii], end=" ")
    elif cfg.lr_config["policy"]=='consine':
        for ii in range(totalEpoches):
            lrList[ii] *= (0.5+0.5*math.cos(math.pi*(ii/totalEpoches)))
    elif cfg.lr_config["policy"]=='restart':
        # not modify here. Modify in each iter instead.
        pass
    elif cfg.lr_config["policy"]=='restart_step':
        lrStages = cfg.lr_config["restartStep"]
        for ii in range(len(lrStages)):
            lrList[lrStages[ii]:]*=0.1
        print("lr adapting stages: ", end=' ')
        for ii in range(len(lrStages)):
            print(cfg.lr_config["step"][ii], end=" ")
    else:
        raise NotImplementedError

    print("Lr: ", lrList)
    print("\ntotal training epoches: ", totalEpoches)

    optimizer_config = cfg.optimizer
    optimizer = optimizer_config['type'](model.parameters(), lr=optimizer_config['lr'], momentum=optimizer_config['momentum'], weight_decay=optimizer_config['weight_decay']) # SGD
    # optimizer = optimizer_config['type'](model.parameters(), lr=optimizer_config['lr'], betas=[0.9, 0.999], weight_decay=optimizer_config['weight_decay']) # Adam
    criterion = nn.CrossEntropyLoss()

    print("starting epoch: ", globalStartEpoch)
    epochSize = len(trainset) // cfg.batchSize  
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
    showIters = 10

    #------------------------------------training--------------------------------
    for epoch in range(leftEpoches):
        trainloader = DataLoader(trainset, batch_size=cfg.batchSize, shuffle=True, pin_memory=True, num_workers=cfg.workers_per_gpu)
        currentEpoch = epoch + pastEpoches

        # 终止训练
        if currentEpoch >= totalEpoches:
            print("Current epoch is larger than setting epoch nums, training stop.")
            return

        # 仅用于打印
        model.train()
        top1.reset()
        top5.reset()
        for batchIdx, (inputs, targets) in enumerate(trainloader):

            iterStartTime = time.time()

            if cfg.lr_config['warmup'] is not None and pastIters < cfg.lr_config['warmup_iters']:
                cur_lr = get_warmup_lr(pastIters, cfg.lr_config['warmup_iters'],
                                        optimizer_config['lr'], cfg.lr_config['warmup_ratio'],
                                        cfg.lr_config['warmup'])
            else:
                if cfg.lr_config["policy"]=='restart' or cfg.lr_config["policy"]=='restart_step':
                    cur_lr = lrList[currentEpoch]*(0.5+0.5*math.cos(batchIdx/len(trainloader)*math.pi))
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
            sys.stdout.flush()

        leftEpoches -= 1
        top1.reset()
        top5.reset()
        save_name = "./weights/" + cfg.name + "_epoch_" + str(currentEpoch) + ".pth"
        torch.save(model.state_dict(), save_name)

        model.eval()
        for batchIdx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                outputs = model(inputs)
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            if batchIdx%int(showIters) == 0 and batchIdx != 0:
                print('validset: Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                    'Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                    top1=top1, top5=top5))

if __name__ == '__main__':
    train(globalStartEpoch=cfg.epoch_iters_start, totalEpoches=cfg.total_epoch)   #设置本次训练的起始epoch
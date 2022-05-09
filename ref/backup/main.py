import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import argparse
from models import *
#from utils import progress_bar
import time

#----------------------解析参数---------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--input_size', default=128, type=int, metavar='N', help='input dimension')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--savedir', default="./checkpoint/modelName")
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='./checkpoint/ckpt.t7')
parser.add_argument('--yourOwnDataset', metavar='DIR', default=True, type=bool, help='your own data')
parser.add_argument('--data', metavar='DIR', default="/home/w/data/256_ObjectCategories", help='path to dataset')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()

#----------------------全局变量---------------------------
start_epoch = 0
best_prec1 = -1
best_prec5 = -1

#----------------------数据---------------------------
print('==> Preparing data..')
inputSize = args.input_size
traindir = '/home/w/data/256_ObjectCategories'
valdir = '/home/w/data/256_ObjectCategories'
transform_train = transforms.Compose([
    transforms.RandomCrop(inputSize, padding=50),
    #transforms.RandomResizedCrop(inputSize, scale=(1, 1), ratio=(1, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.Resize([inputSize, inputSize]),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.yourOwnDataset:
    trainset = datasets.ImageFolder(traindir, transform=transform_train)
    testset = datasets.ImageFolder(valdir, transform=transform_test)
else:
    trainset = torchvision.datasets.CIFAR10(root=traindir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=valdir, train=False, download=True, transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

#----------------------网络---------------------------
print('==> Building model..')
#net = VGG('VGG11')
net = VoVNet('57')
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()

if torch.cuda.is_available():
    device = 'cuda'
    net.cuda()
    # parallel use GPU
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

#----------------------模型存储---------------------------
savedir = args.savedir
if not os.path.exists(args.savedir):
    os.makedirs(savedir)

log_path = savedir+"/training_log.txt"
if not os.path.exists(log_path):
    with open(log_path, "a") as myfile:
        myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tpred@1\t\tpred@5\t\tlearningRate")

print('==> Resuming from checkpoint..')
if args.resume:
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_prec1 = checkpoint['best_prec1']
    best_prec5 = checkpoint['best_prec5']
    start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("the current best_prec1 is ", best_prec1)
    print("the current best_prec5 is ", best_prec5)
else:
    print("No resuming.")


# 一个epoch
def train(epoch):
    net.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for batchIdx, (inputs, targets) in enumerate(trainloader):
        data_time.update(time.time() - end)
        
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        lr = 0
        if batchIdx % args.print_freq == 0:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print('Epoch: [{0}][{1}/{2}][lr:{lr:.6g}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.2f}) / '
                  'Data {data_time.val:.3f} ({data_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                   epoch, batchIdx, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr))
    return losses

def test(epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    net.eval()

    end = time.time()
    with torch.no_grad():
        for batchIdx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if batchIdx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    batchIdx, len(testloader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg, top5.avg, losses
    '''
    acc = 100.*correct/total

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'best_prec5': 0
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    '''

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed
    by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

for epoch in range(start_epoch, start_epoch+1000):
    train_losses = train(epoch)
    prec1, prec5, val_losses = test(epoch)

    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    filenameCheckpoint = savedir + '/checkpoint.pth.tar'
    filenameBest = savedir + '/model_best.pth.tar'.format(epoch+1, prec1, prec5)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'best_prec1': best_prec1,
        'best_prec5': best_prec5,
        'optimizer' : optimizer.state_dict(),
    }, is_best, filenameCheckpoint, filenameBest)

    if is_best:
        with open(savedir+"/best.txt", "w") as myfile:
            myfile.write("Best accuracy : (%.4f, %.4f) at %d epoch" 
                % (prec1, prec5, epoch))
    #save log
    LR = 0
    for param_group in optimizer.param_groups:
        LR = float(param_group['lr'])

    with open(log_path, "a") as myfile:
        myfile.write("\n%d\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\t\t%.8f"
            % (epoch, train_losses.avg, val_losses.avg, prec1, prec5, LR))

    torch.cuda.empty_cache()
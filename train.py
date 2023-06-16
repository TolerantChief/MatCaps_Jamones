""""
Instrucciones:

En la función get_setting indicar la dirección de nuestro set de entrenamiento y test en las variables train_path y test_path.
En la función main(), en el apartado model hay que elegir la configuración de MatCaps que queremos. Hay que dejar una sin comentar y comentar el resto.
En Training settings un poco más abajo se pueden ver todos los parámetros que se pueden modificar, como el batch size, learning rate etc.

"""



from __future__ import print_function
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchbearer.callbacks import EarlyStopping
from mem_profile import get_gpu_memory_map
from numpy import prod
import matplotlib.pyplot as plt


import torchvision

from model import capsules
from loss import SpreadLoss

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Matrix-Capsules-EM')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for testing (default: 20)')
parser.add_argument('--test-intvl', type=int, default=1, metavar='N',
                    help='test intvl (default: 1)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 25)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight-decay', type=float, default=0, metavar='WD',
                    help='weight decay (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--em-iters', type=int, default=2, metavar='N',
                    help='iterations of EM Routing')
parser.add_argument('--snapshot-folder', type=str, default='./snapshots', metavar='SF',
                    help='where to store the snapshots')
parser.add_argument('--data_folder', type=str, default='/content/data', metavar='DF',
                    help='where to store the datasets')
parser.add_argument('--dataset', type=str, default='JAMONES_CROPPED', metavar='D',
                    help='dataset for training')


def get_setting(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    #Indicar dirección del set de entrenamiento y test.
    data_path = os.path.join(args.data_folder, args.dataset)

    size = 100
    mean, std = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([
    # shift by 2 pixels in either direction with zero padding.
    transforms.Resize((size,size)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomRotation(10),
    transforms.RandomCrop(size, padding=2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

    test_size = 0.2
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
    num_data = len(dataset)
    num_test = int(test_size * num_data)
    num_train = num_data - num_test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # train_dataset = torchvision.datasets.ImageFolder(
    #     root=data_path,
    #     transform=transform
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True, **kwargs
    # )
    # test_dataset = torchvision.datasets.ImageFolder(
    #     root=test_path,
    #     transform=transform
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True, **kwargs
    # )
    print(test_loader)
    print("Detected Classes are: ", dataset.class_to_idx)
    num_class=26

    return num_class,train_loader,test_loader


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
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


def train(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    train_len = len(train_loader)
    epoch_acc = 0
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        r = (1.*batch_idx + (epoch-1)*train_len) / (args.epochs*train_len)
        loss = criterion(output, target, r)
        acc = accuracy(output, target)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        epoch_acc += acc[0].item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}\t[{}/{} ({:.0f}%)]\t'
                  'Loss: {:.6f}\tAccuracy: {:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader),
                  loss.item(), acc[0].item(),
                  batch_time=batch_time, data_time=data_time))
    return epoch_acc, loss.item()


def snapshot(model, folder, epoch):
    path = os.path.join(folder, 'model_{}.pth'.format(epoch))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print('saving model to {}'.format(path))
    torch.save(model.state_dict(), path)


def test(test_loader, model, criterion, device):
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, r=1).item()
            acc += accuracy(output, target)[0].item()

    test_loss /= test_len
    acc /= test_len
    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, acc))
    with open("log.txt", "a") as f:
     f.write('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f} \n'.format(
        test_loss, acc))
    return acc, test_loss 


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    open('log.txt', 'w').close()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    # datasets
    num_class, train_loader, test_loader = get_setting(args)

    # model
    # A, B, C, D = 64, 8, 16, 16
    # A, B, C, D = 32, 32, 32, 32
    A, B, C, D = 32, 4, 4, 4    

    model = capsules(A=A, B=B, C=C, D=D, E=num_class,
                     iters=args.em_iters).to(device)

    print('Num params:', sum([prod(p.size())
              for p in model.parameters()]))

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)

    max_memory_usage = 0

    early_stopping = EarlyStopping(patience=10)
    best_epoch = 0
    epochs_no_improve = 0

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    best_acc, _ = test(test_loader, model, criterion, device)
    for epoch in range(1, args.epochs + 1):
        acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, device)
        acc /= len(train_loader)
        history['train_acc'].append(acc)
        history['train_loss'].append(train_loss)
        scheduler.step(acc)
        if epoch % args.test_intvl == 0:
            test_acc, test_loss = test(test_loader, model, criterion, device)
            history['test_acc'].append(test_acc)
            history['test_loss'].append(test_loss)
            best_acc = max(best_acc, test_acc)

            if test_acc > best_acc:
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == early_stopping.patience:
                    print('#### EARLY STOPPING ####')
                    current_memory_usage = get_gpu_memory_map()[0]
                    if current_memory_usage > max_memory_usage:
                        max_memory_usage = current_memory_usage

                    best_acc = max(best_acc, test(test_loader, model, criterion, device))
                    print('best test accuracy: {:.6f} in epoch: {}'.format(best_acc, best_epoch))

                    print('max. memory usage: ', max_memory_usage)

                    snapshot(model, args.snapshot_folder, args.epochs)

                    epochs = range(1, epoch + 1)
                            
                    plt.figure()
                        
                    plt.subplot(2, 1, 1)
                    
                    plt.plot(epochs, history['train_acc'], label='Training Accuracy')
                    plt.plot(epochs, history['test_acc'], label='Validation Accuracy')
                    plt.xlabel('Epochs')
                    plt.ylabel('Accuracy')
                    plt.title('Training and Validation Accuracy')
                    plt.legend()
                        
                    plt.subplot(2,1,2)
                        
                    plt.plot(epochs, history['train_loss'], label='Training Loss')
                    plt.plot(epochs, history['test_loss'], label='Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss')
                    plt.legend()
                            
                    plt.tight_layout()
                    plt.show()  
                    return

    current_memory_usage = get_gpu_memory_map()[0]
    if current_memory_usage > max_memory_usage:
        max_memory_usage = current_memory_usage

    best_acc, _ = max(best_acc, test(test_loader, model, criterion, device))
    print('best test accuracy: {:.6f}'.format(best_acc))

    print('max. memory usage: ', max_memory_usage)

    snapshot(model, args.snapshot_folder, args.epochs)

if __name__ == '__main__':
    main()

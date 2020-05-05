from __future__ import print_function

import copy
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import wandb
import matplotlib.pyplot as plt

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.utils.weight_norm as WeightNormt

import torch.utils.data.sampler as sampler



def conv_block(in_c,out_c,args):
    if args.norm == 5:
        return nn.Sequential(
            WeightNorm(nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1)),
            nn.ReLU(),

            WeightNorm(nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1)),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
    elif args.norm < 5:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
            Normalization(type=args.norm, channels=out_c),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1),
            Normalization(type=args.norm, channels=out_c),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )






class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv_1 = conv_block(in_c=3,out_c=32,args=args)
        self.conv_2 = conv_block(in_c=32,out_c=64,args=args)
        self.conv_3 = conv_block(in_c=64, out_c=128,args=args)
        self.fc_layer = nn.Sequential(nn.Linear(2048,128),
                                                nn.ReLU(),
                                                nn.Dropout(0.2),
                                                nn.Linear(128,10),
                                                nn.LogSoftmax())
    def forward(self, x):
        #conv layers
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        #fc layer
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')


def Normalization(type, channels):
    if type == 1:
        return nn.BatchNorm2d(num_features=channels)
    elif type == 2:
        return nn.GroupNorm(num_channels=channels, num_groups=int(channels / 4))
    elif type == 3:
        return nn.InstanceNorm2d(num_features=channels)
    elif type == 4:
        return nn.GroupNorm(num_channels=channels, num_groups=1)


def train(args, model, device, train_loader, optimizer, epoch):
    print('Optimizing epoch {}'.format(epoch))
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        #loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        #if batch_idx % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #               100. * batch_idx / len(train_loader), loss.item()))
        #    train_losses.append(loss.item())
        #    train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

def train_model(args, model, device, train_loader, test_loader, valid_loader, optimizer, epoch):
    best_epoch = 0
    best_accuracy = 0
    best_loss = 100
    counter = 0 #Used for early stopping
    for epoch in range(1, args.epochs + 1):
        print('Optimizing Epoch: {}...'.format(epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output,target)
            loss.backward()
            optimizer.step()

        train_stats = evaluate_model(args, model, device, train_loader)
        valid_stats = evaluate_model(args, model, device, valid_loader)
        test_stats = evaluate_model(args, model, device, test_loader)

        print(
            "\n Train Accuracy: {}%, Train Loss: {} \n Validation Accuracy:{}%, Validation Loss: {} \n Test Accuracy: {}%, Test Loss: {} \n ".format(
                train_stats[0], train_stats[1], valid_stats[0], valid_stats[1], test_stats[0], test_stats[1]))
        wandb.log({"Train_Accuracy": train_stats[0], "Train_Loss": train_stats[1], "Validation_Accuracy":valid_stats[0],
                   "Validation_Loss": valid_stats[1], "Test_Accuracy": test_stats[0], "Test_Loss": test_stats[1]})

        if test_stats[0] > best_accuracy:
            best_accuracy = test_stats[0]
            best_epoch = epoch

        if valid_stats[1] <= best_loss:
            best_loss = valid_stats[1]
        elif args.early_stopping == 1:
            counter += 1
            if counter > args.patience:
                print("Early stopping")
                return best_accuracy, best_epoch



    return best_accuracy, best_epoch





train_losses = []
train_counter = []
test_losses = []
test_counter = []
highest_accuracy = 0


def test_func():
    return 1


def evaluate_model(args, model, device, data_loader):
    correct = 0
    loss = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss += F.nll_loss(output, target, reduction='sum').item()
        loss /= len(data_loader.dataset)
        return round(correct/len(data_loader.dataset) * 100., 3), round(loss,3)





def main():
    wandb.init(project="VGG_CIFAR10_Validation")

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.0005  )')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--norm', type=int, default=1, metavar='NT',
                        help='1-BatchNorm, 2-GroupNorm, 3-InstanceNorm, 4-LayerNorm, 5-WeightNorm, 6-None')
    parser.add_argument('--optimizer',type=int,default=1,metavar='OPT',
                        help='1-SGD, 2-Adam')
    parser.add_argument('--cuda', type=int, default=0, metavar='CUDA',
                        help='which GPU to run on')
    parser.add_argument('--valid-size', type=int, default=5000, metavar='VS',
                        help='how large should the validation set be?')
    parser.add_argument('--early-stopping', type=int, default=1, metavar='ES',
                        help='use early stopping')
    parser.add_argument('--patience', type=int, default=10, metavar='P',
                        help='patience setting for early stopping')
    args = parser.parse_args()

    def norm_type(i):
        switcher={
            1:'BatchNorm',
            2:'GroupNorm',
            3:'InstanceNorm',
            4:'LayerNorm',
            5:'WeightNorm',
            6:'None'
        }
        return switcher.get(i,"--Invalid Normalization")
    def optim_type(i):
        switcher={
            1:'SGD',
            2:'Adam'
        }
        return switcher.get(i,"--Invalid Optimizer")

    wandb.run.name = str(optim_type(args.optimizer))+"-lr:"+str(args.learning_rate)+"-bs:"+str(args.batch_size)+"+"+str(norm_type(args.norm))
    wandb.run.save()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    wandb.config.update(args)

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda:" + str(args.cuda))
    else:
        device = torch.device("cpu")

    kwargs = {'num_workers': 12, 'pin_memory': True} if use_cuda else {}


    valid_set_size = args.valid_size


    # output of torchvision datasets ar PILImage images of rand [0,1]. Transform them to Tensors of normalized range [-1,1].
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )


    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    valid_set = copy.deepcopy(train_set)

    train_set.data = train_set.data[:(len(train_set)-valid_set_size)]
    valid_set.data = valid_set.data[(len(valid_set)-valid_set_size):]

    train_set.targets = train_set.targets[:(len(train_set.targets)-valid_set_size)]
    valid_set.targets = valid_set.targets[(len(valid_set.targets)-valid_set_size):]

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, **kwargs)



    # Training
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = Net(args=args).to(device)

    if args.optimizer == 1:
        #model.apply(init_weights)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    wandb.watch(model)

#Call Training Loop
    stats = train_model(args, model, device, train_loader, test_loader, valid_loader, optimizer, args.epochs)
    print("\n Best Accuracy: {}%, Best Epoch: {}".format(stats[0], stats[1]))
    wandb.log({'Best_Accuracy':stats[0],'Best_Epoch':stats[1]})

    #for epoch in range(1, args.epochs + 1):
     #   print("Training Epoch {}".format(epoch))
      #  stats = train_model(args, model, device, train_loader, test_loader, valid_loader, optimizer, epoch)
       # #print("Training Epoch {}".format(epoch))
        #stats = train_model(args, model, device, train_loader, test_loader, valid_loader, optimizer,epoch)
        #print("\n Train Accuracy: {}%, Train Loss: {} \n Validation Accuracy:{}%, Validation Loss: {} \n Test Accuracy: {}%, Test Loss: {} \n ".format(stats[0][0], stats[0][1], stats[1][0], stats[1][1], stats[2][0], stats[2][1]))
        #wandb.log({"Train_Accuracy":stats[0][0],"Train_Loss":stats[0][1], "Validation_Accuracy":stats[1][0], "Validation_Loss":stats[1][1], "Test_Accuracy":stats[2][0], "Test_Loss":stats[2][1]})
        #train(args, model, device, train_loader, optimizer, epoch)
        #validate(args, model, device, valid_loader)
        #test(args, model, device, test_loader=test_loader)


if __name__ == '__main__':
    main()
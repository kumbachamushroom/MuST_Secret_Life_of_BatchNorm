from __future__ import print_function
import argparse
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import wandb
import matplotlib.pyplot as plt
import torch.nn.utils.weight_norm as WeightNorm


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(28 * 28, 784),
            Normalization(args.norm, 784),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(784, 784),
            Normalization(args.norm, 784),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(784, 784),
            Normalization(args.norm, 784),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(784, 392),
            Normalization(args.norm, 392),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(392, 10),
            Normalization(args.norm, 10),
            nn.LogSoftmax(),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


def Normalization(type, channels):
    if type == 1:
        return nn.BatchNorm1d(num_features=channels)
    elif type == 2:
        return nn.LayerNorm(channels)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
    if type(m) == nn.Conv2d:
        torch.nn.init.orthogonal_(m.weight)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        #  example_images.append(wandb.Image(
        #  data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    wandb.log({
        "Test_Accuracy": 100. * correct / len(test_loader.dataset),
        "Test_Loss": test_loss})


train_losses = []
train_counter = []
test_losses = []
test_counter = []


def main():
    wandb.init(project="MNIST_MLP")

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
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
    parser.add_argument('--optimizer', type=int, default=1, metavar='OPT',
                        help='1-SGD, 2-Adam')
    parser.add_argument('--cuda', type=int, default=0, metavar='CUDA',
                        help='which GPU to run on')
    args = parser.parse_args()

    def norm_type(i):
        switcher = {
            1: 'BatchNorm',
            2: 'LayerNorm',
            3: 'InstanceNorm',
            4: 'LayerNorm',
            5: 'WeightNorm',
            6: 'None',
            7: 'OrthInit'
        }
        return switcher.get(i, "--Invalid Normalization")

    def optim_type(i):
        switcher = {
            1: 'SGD',
            2: 'Adam'
        }
        return switcher.get(i, "--Invalid Optimizer")

    wandb.run.name = "MLP-Deep-" + str(optim_type(args.optimizer)) + "-lr:" + str(args.lr) + "-bs:" + str(
        args.batch_size) + "+" + str(norm_type(args.norm))
    wandb.run.save()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    wandb.config.update(args)

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda:" + str(args.cuda))
    else:
        device = torch.device("cpu")

    kwargs = {'num_workers': 6, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_counter = [i * len(train_loader.dataset) for i in range(args.epochs + 1)]

    model = Net(args).to(device)
    if args.optimizer == 1:
        if args.norm == 7:
            model.apply(init_weights)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    wandb.watch(model)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    # plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    # plt.show()
    wandb.log({"losses": plt})


if __name__ == '__main__':
    main()



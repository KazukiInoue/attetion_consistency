import cv2
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

modelpath = './net.pth'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16*5*5, 10)
        self.feature = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        self.feature = x
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc(x))

        return x


def train(net):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        runnnig_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runnnig_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d], loss: %.3f' %
                      (epoch+1, i+1, runnnig_loss/2000))
                runnnig_loss = 0.0

    print('Finished Training')
    torch.save({'model': net.state_dict()}, modelpath)
    return net


def test(net):
    checkpoint = torch.load(modelpath)
    net.load_state_dict(checkpoint['model'])
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images = data[0].to(device)
            labels = data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total
        ))


def grad_cam(net, filepath):

    img_np = Image.open(filepath)
    img_size = (32, 32)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    img = transform(img_np)
    img = img.view(1, 3, img_size[0], img_size[1])

    print(img.shape)

    checkpoint = torch.load(modelpath)
    net.load_state_dict(checkpoint['model'])

    outputs = net(img)


if __name__ == '__main__':
    net = Net()
    net.to(device)

    # net = train(net)

    # test(net)

    filepath = './cat.jpeg'
    grad_cam(net, filepath)

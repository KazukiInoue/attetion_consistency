import cv2
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import vgg
from models.net import Net

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

bsize = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bsize,
                                         shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

modelpath = './checkpoints/normal_vgg.pth'


def train(net):

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

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


def cam(net):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    with torch.no_grad():
        for data in testloader:
            images = data[0].to(device)
            bsize = images.size()[0]
            image_size = images.size()[2:]

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            weights = net.fc.weight.data

            feature_map = net.feature_map
            nchannel = feature_map.size()[1]

            target_weights = weights[predicted[0], :].view(1, -1)

            for i in range(1, bsize):
                tmp = weights[predicted[i]].view(1, -1)
                target_weights = torch.cat([target_weights, tmp], 0)

            masks = torch.empty_like(feature_map)
            for b in range(bsize):
                for c in range(nchannel):
                    masks[b, c, :, :] = target_weights[b, c] * \
                        feature_map[b, c, :, :]

            masks = torch.sum(masks, 1)
            masks = F.adaptive_avg_pool2d(masks, image_size)

            for i in range(bsize):
                image = images[i].data.cpu().numpy()
                mask = masks[i].data.cpu().numpy()

                mask = mask - np.min(mask)
                if np.max(mask) != 0:
                    mask = mask / np.max(mask)

                mask = np.float32(cv2.applyColorMap(
                    np.uint8(255*mask), cv2.COLORMAP_JET))
                cam = mask + \
                    np.float32((np.uint8(image.transpose((1, 2, 0))*255)))

                cam = cam - np.min(cam)
                if np.max(cam) != 0:
                    cam = cam / np.max(cam)

                cam = np.uint8(255 * cam)
                cv2.imwrite(str(i)+'.jpg', cam)

            exit()


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    net = vgg.modified_vgg16(num_classes=10)
    # net = Net()
    net.to(device)

    net = train(net)
    test(net)

    # cam(net)

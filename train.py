import cv2
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from models import vgg
from models.net import AttNet, Net
from options.train_options import TrainOptions
from options.test_options import TestOptions


if __name__ == '__main__':

    opt = TrainOptions().parse

    # net = vgg.modified_vgg16(num_classes=10)
    net = vgg.vgg16(num_classes=10)
    # net = AttNet(num_classes=10)
    # net = Net(num_classes=10)

    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                            shuffle=False, num_workers=2)

    logger = SummaryWriter(opt.log_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    cls_criterion = nn.CrossEntropyLoss()

   for epoch in range(num_epoch):
        runnnig_loss = 0.0
        for step, data in enumerate(trainloader, 0):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            cls_loss = cls_criterion(outputs, labels)
            loss = cls_loss
            info = {'cls_loss': cls_loss}

            if training_type == 'hflip' or 'att_consist':
                inputs_hflip = inputs.clone()
                inputs_hflip = inputs_hflip[:, :, :,
                                            torch.arange(inputs.size()[3]-1, -1, -1)]
                outputs_hflip = net(inputs_hflip)
                _, preditcted_hflip = torch.max(outputs_hflip.data, 1)

                cls_hflip_loss = cls_criterion(outputs_hflip, labels)
                loss += cls_hflip_loss

                info['cls_hflip_loss'] = cls_hflip_loss

            if training_type == 'att_consist':
                masks = net.cam(predicted)
                masks_hflip = net.cam(predicted)

                masks_hflip = masks_hflip[:, :, torch.arange(
                    masks.size()[2]-1, -1, -1)]

                mask_criterion = nn.MSELoss()
                mask_loss = mask_criterion(masks, masks_hflip)

                loss = loss + mask_loss

                info['mask_loss'] = mask_loss

            loss.backward()
            optimizer.step()

            runnnig_loss += loss.item()
            if step % display_count == display_count-1:
                print('[%d, %5d], loss: %.3f' %
                      (epoch+1, step+1, runnnig_loss/display_count))
                runnnig_loss = 0.0
            if step % 100 == 0:
                logger.add_scalars("logs_s_{}/losses".format(training_type),
                                   info, epoch * (len(trainset)/bsize) + step)

    print('Finished Training')
    torch.save({'model': net.state_dict()}, modelpath)
    logger.close()
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

    checkpoint = torch.load(modelpath)
    net.load_state_dict(checkpoint['model'])

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

                img_np = np.float32((np.uint8(image.transpose((1, 2, 0))*255)))
                cam = np.uint8(255 * cam)
                cv2.imwrite(str(i)+'.jpg', cam)
                cv2.imwrite(str(i)+'_org.jpg', img_np)

            exit()


if __name__ == '__main__':

    net.to(device)

    net = train(net)
    test(net)

    # cam(net)

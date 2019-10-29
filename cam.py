import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms


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

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from models.classification_model import ClassificationModel
from options.train_options import TrainOptions

from data.dataloader_dataset import DataloaderDataset

if __name__ == '__main__':

    opt = TrainOptions().parse()

    dataloader_dataset = DataloaderDataset(opt)
    dataloader = dataloader_dataset.dataloader
    dataset_size = len(dataloader_dataset.dataset)

    model = ClassificationModel(opt)

    logger = SummaryWriter(opt.log_dir)
    total_steps = 0

    for epoch in range(opt.n_epochs):
        steps_in_epoch = 0
        for step, data in enumerate(dataloader, 0):
            model.update(data)
            total_steps += opt.batch_size
            steps_in_epoch += opt.batch_size

            if total_steps % opt.print_freq == 0:
                print('epoch %d, iteration %d/%d loss %.4f' %
                      (epoch, steps_in_epoch, dataset_size, model.loss))
                logger.add_scalar('%s %s %s  losses' % (
                    opt.name, opt.net, opt.dataset), model.loss, total_steps)
                logger.close()

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, iteration %d/%d)' %
                      (epoch, steps_in_epoch, dataset_size))
                model.save('latest')

            # optimizer.zero_grad()
            # outputs = net(inputs)

            # _, predicted = torch.max(outputs.data, 1)
            # cls_loss = cls_criterion(outputs, labels)

            # if training_type == 'hflip' or 'att_consist':
            #     inputs_hflip = inputs.clone()
            #     inputs_hflip = inputs_hflip[:, :, :,
            #                                 torch.arange(inputs.size()[3]-1, -1, -1)]
            #     outputs_hflip = net(inputs_hflip)
            #     _, preditcted_hflip = torch.max(outputs_hflip.data, 1)

            #     cls_hflip_loss = cls_criterion(outputs_hflip, labels)
            #     loss += cls_hflip_loss

            #     info['cls_hflip_loss'] = cls_hflip_loss

            # if training_type == 'att_consist':
            #     masks = net.cam(predicted)
            #     masks_hflip = net.cam(predicted)

            #     masks_hflip = masks_hflip[:, :, torch.arange(
            #         masks.size()[2]-1, -1, -1)]

            #     mask_criterion = nn.MSELoss()
            #     mask_loss = mask_criterion(masks, masks_hflip)

            #     loss = loss + mask_loss

            #     info['mask_loss'] = mask_loss

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, total_steps %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch+1)


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
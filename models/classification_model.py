import os

import torch
import torch.optim as optim
from torch import nn

from models import vgg
from models.net import AttNet, Net
from models.vgg_att_net import VGGAttNet


class ClassificationModel():

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        if opt.net == 'vgg':
            self.net = vgg.vgg16(num_classes=opt.n_classes)
        elif opt.net == 'normal':
            self.net = Net(num_classes=opt.n_classes)
        elif opt.net == 'att':
            self.net = AttNet(num_classes=opt.n_classes)
        elif opt.net == 'vgg_att':
            self.net = VGGAttNet(num_classes=opt.n_classes)
        else:
            raise ValueError('[%s] cannot be used!' % opt.net)

        self.net = self.net.to(self.device)

        if not opt.is_train:
            self.load_network(self.net, self.opt.which_epoch)
        else:
            if self.opt.resume_train:
                self.load_network(self.opt.which_epoch)

            self.cls_criterion = nn.CrossEntropyLoss()
            if opt.training_type == 'att_consist':
                self.mask_criterion = nn.MSELoss()
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=0.001, momentum=0.9)

    def forward(self):
        self.outputs = self.net(self.images)
        _, self.predicted = torch.max(self.outputs, 1)

        if self.opt.training_type == 'hflip' or 'att_consist':
            hflip = self.images.clone()
            hflip = hflip[:, :, :,
                          torch.arange(self.images.size()[3]-1, -1, -1)]
            self.outputs_hflip = self.net(hflip)
            _, self.predicted_hflip = torch.max(self.outputs_hflip, 1)

            if self.opt.training_type == 'att_consist':
                self.masks = self.cam(self.predicted)
                masks_hflip = self.cam(self.predicted_hflip)
                self.masks_hflip = masks_hflip[:, :, torch.arange(
                    self.masks.size()[2]-1, -1, -1)]

    def backward(self):
        self.loss = self.cls_criterion(self.outputs, self.labels)
        if self.opt.training_type == 'hflip' or 'att_consist':
            self.loss += self.cls_criterion(self.outputs_hflip, self.labels)
            if self.opt.training_type == 'att_consist':
                self.loss += self.mask_criterion(self.masks, self.masks_hflip)

        self.loss.backward()

    def update(self, data):
        self.images = data[0].to(self.device)
        self.labels = data[1].to(self.device)

        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    def cam(self, predicted):
        weights = self.net.classifier.weight

        feature_map = self.net.feature_map

        target_weights = weights[predicted[0]].view(1, -1)
        for i in range(1, self.opt.batch_size):
            tmp = weights[predicted[i]].view(1, -1)
            target_weights = torch.cat([target_weights, tmp], 0)

        # target_weights = target_weights.unsqueeze(0).unsqueeze(1)

        # feature_map = feature_map.transpose(0, 2)
        # feature_map = feature_map.transpose(1, 3)

        # masks = torch.mul(feature_map, target_weights)
        # masks = masks.transpose(0, 2)
        # masks = masks.transpose(1, 3)

        # masks = torch.sum(masks, dim=1)

        target_weights_mat = target_weights.unsqueeze(2).unsqueeze(3)
        target_weights_mat = target_weights.repeat(
            1, 1, feature_map.size()[2], feature_map.size()[3])

        masks = torch.mul(feature_map, target_weights_mat)
        masks = torch.sum(masks, dim=1)

        return masks

    def test(self, data):
        self.images = data[0].to(self.device)
        self.labels = data[1].to(self.device)
        self.forward()
        self.correct = (self.predicted == self.labels).sum()

    def save(self, epoch_label):
        save_filename = '%s_%s_net.pth' % (epoch_label, self.opt.net)
        save_path = os.path.join(self.opt.save_dir, save_filename)
        torch.save(self.net.to('cpu').state_dict(), save_path)
        self.net.to(self.device)

    def load_network(self, epoch_label):
        save_filename = '%s_%s_net.pth' % (epoch_label, self.opt.net)
        save_path = os.path.join(self.opt.save_dir, save_filename)
        self.net.load_state_dict(torch.load(save_path))

        if not self.opt.is_train:
            print('%s are loaded' % save_path)

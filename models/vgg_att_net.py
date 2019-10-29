import torch
import torch.nn as nn


class VGGAttNet(nn.Module):

    def __init__(self, num_classes=1000, batch_norm=False, init_weights=True):
        super(VGGAttNet, self).__init__()
        if init_weights:
            self._initialize_weights()

        cfg = (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 1024, 'A')

        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'A':
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        self.convs = nn.Sequential(*layers)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.convs(x)
        self.feature_map = x.clone()
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def cam(self, predicted):

        weights = self.classifier.weight

        feature_map = self.feature_map
        bsize = feature_map.size()[0]

        target_weights = weights[predicted[0]].view(1, -1)

        for i in range(1, bsize):
            tmp = weights[predicted[i]].view(1, -1)
            target_weights = torch.cat([target_weights, tmp], 0)

        target_weights = target_weights.unsqueeze(0).unsqueeze(1)

        feature_map = feature_map.transpose(0, 2)
        feature_map = feature_map.transpose(1, 3)

        masks = torch.mul(feature_map, target_weights)
        masks = masks.transpose(0, 2)
        masks = masks.transpose(1, 3)

        masks = torch.sum(masks, dim=1)

        return masks
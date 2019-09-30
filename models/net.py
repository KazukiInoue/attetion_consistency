import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.conv4 = nn.Conv2d(16, 16, 3)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)
        self.fmap = None

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.max_pool(F.relu(self.conv3(x)))
        x = self.max_pool(F.relu(self.conv4(x)))

        self.feature_map = x
        x = self.avg_pool(x)
        x = x.view(-1, 16)
        x = self.fc(F.relu(x))

        return x

    def cam(self, predicted):

        weights = self.fc.weight.data

        feature_map = self.feature_map
        bsize, nchannel = feature_map.size()[0], feature_map.size()[1]

        target_weights = weights[predicted[0], :].view(1, -1)

        for i in range(1, bsize):
            tmp = weights[predicted[i]].view(1, -1)
            target_weights = torch.cat([target_weights, tmp], 0)

        masks = torch.empty_like(feature_map)
        for b in range(bsize):
            for c in range(nchannel):
                masks[b, c, :, :] = target_weights[b, c] * \
                    feature_map[b, c, :, :]

        masks = torch.sum(masks, dim=1)

        return masks

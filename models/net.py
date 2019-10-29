import torch
import torch.nn as nn
import torch.nn.functional as F


class AttNet(nn.Module):
    def __init__(self, num_classes):
        super(AttNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.conv2 = nn.Conv2d(8, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.conv4 = nn.Conv2d(16, 16, 3)
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.conv6 = nn.Conv2d(16, 16, 3)

        self.max_pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)
        self.fmap = None

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.max_pool(F.relu(self.conv3(x)))
        x = self.max_pool(F.relu(self.conv4(x)))
        x = self.max_pool(F.relu(self.conv5(x)))
        x = self.max_pool(F.relu(self.conv6(x)))

        self.feature_map = x
        print('in forward', x.size())
        x = self.avg_pool(x)
        x = x.view(-1, 16)
        x = self.fc(F.relu(x))

        return x

    def cam(self, predicted):

        weights = self.fc.weight
        print('weights.size()', weights.size())

        feature_map = self.feature_map
        feature_map = feature_map
        bsize = feature_map.size()[0]

        target_weights = weights[predicted[0], :].view(1, -1)

        for i in range(1, bsize):
            tmp = weights[predicted[i]].view(1, -1)
            target_weights = torch.cat([target_weights, tmp], 0)

        target_weights = target_weights.unsqueeze(0).unsqueeze(1)

        feature_map = feature_map.transpose(0, 2)
        feature_map = feature_map.transpose(1, 3)

        print(feature_map.size())
        print(target_weights.size())
        exit()

        masks = torch.mul(feature_map, target_weights)
        masks = masks.transpose(0, 2)
        masks = masks.transpose(1, 3)

        masks = torch.sum(masks, dim=1)

        return masks


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

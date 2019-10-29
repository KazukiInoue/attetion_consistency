import torch.utils.data
import torchvision
from torchvision import transforms


class DataloaderDataset():

    def __init__(self, opt):
        self.opt = opt
        transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if opt.dataset == 'CIFAR10':
            self.dataset = torchvision.datasets.CIFAR10(
                root='./dataset', train=True,
                download=True, transform=transform)
        elif opt.dataset == 'CIFAR100':
            self.dataset = torchvision.datasets.CIFAR100(
                root='./dataset', train=True,
                download=True, transform=transform)
        else:
            raise ValueError('[%s] cannot be used!' % opt.dataset)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=opt.batch_size,
            shuffle=True, num_workers=2)

        print('%d images are loaded' % len(self.dataset))

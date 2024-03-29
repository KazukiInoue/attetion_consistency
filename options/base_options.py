import argparse
import os


class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.parser = argparse.ArgumentParser(
            description='Train a Classification Network',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self, parser):
        parser.add_argument_group('base options')
        parser.add_argument(
            '--dataset', type=str,
            default='CIFAR10', help='[CIFAR10 | CIFAR100]')
        parser.add_argument(
            '--net', type=str,
            default='vgg', help='[vgg | normal | att | vgg_att]'
        )
        parser.add_argument(
            '--img_size', type=int,
            default=128, help='resize image',)
        parser.add_argument(
            '--batch_size', type=int,
            default=32, help='batch size')

        parser.add_argument(
            '--checkpoints_dir', type=str,
            default='./checkpoints',
            help='models are saved here')
        parser.add_argument(
            '--name', type=str,
            default='experiment_name',
            help='name of the experiment.')
        parser.add_argument(
            '--which_epoch', type=str,
            default=0,
            help='which epoch to load? set to "latest" to use latest model.')

        self.initialized = True

        return parser

    def parse(self):
        if not self.initialized:
            parser = self.initialize(self.parser)

        opt = parser.parse_args()
        opt.is_train = self.is_train

        opt.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        if opt.dataset == 'CIFAR10':
            opt.n_classes = 10
        elif opt.dataset == 'CIFAR100':
            opt.n_classes = 100
        else:
            raise ValueError('[%s] cannot be used!' % opt.dataset)

        if opt.is_train:
            if not opt.resume_train:
                if os.path.exists(opt.save_dir):
                    raise ValueError(
                        'experiment [%s] are already done!' % opt.name)

                opt.log_dir = os.path.join(opt.log_root, opt.name)
                os.makedirs(opt.save_dir)
                opt.which_epoch = 0

        args = vars(opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk
        save_path = os.path.join(opt.save_dir, 'opt.txt')
        with open(save_path, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        return opt

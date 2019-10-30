from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument_group('train options')
        parser.add_argument(
            '--n_epochs', type=int,
            default=10,  help='number of epochs to train')
        parser.add_argument(
            '--print_freq',  type=int,
            default=200,
            help='frequency of showing training results on console')
        parser.add_argument(
            '--save_latest_freq', type=int,
            default=10000, help='frequency of saving the latest results')
        parser.add_argument(
            '--save_epoch_freq', type=int,
            default=5,
            help='frequency of saving checkpoints at the end of epochs')

        parser.add_argument(
            '--training_type', type=str,
            default='normal',
            help='[normal | hflip | att_consist]'
        )
        parser.add_argument(
            '--log_root', type=str,
            default='./logs',
            help='directory to save training log')

        # resume trained model
        parser.add_argument(
            '--resume_train', type=bool,
            default=False,
            help='resume training at checkpoint')

        self.is_train = True

        return parser

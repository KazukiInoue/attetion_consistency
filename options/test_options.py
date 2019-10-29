from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument_group('test options')
        parser.add_argument(
            '--load_dir', type=str,
            required=True,
            help='directory to load models')
        parser.add_argument(
            '--model_file', type=str,
            required=True,
            help='trained model file name', )

        self.is_train = False

        return parser

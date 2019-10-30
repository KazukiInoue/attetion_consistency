from data.dataloader_dataset import DataloaderDataset
from models.classification_model import ClassificationModel
from options.test_options import TestOptions

if __name__ == '__main__':

    opt = TestOptions().parse()

    print('dataset: ', opt.dataset)
    print('model: ', opt.save_dir)
    print('which epoch: ', opt.which_epoch)

    dataloader_dataset = DataloaderDataset(opt)
    dataloader = dataloader_dataset.dataloader
    dataset_size = len(dataloader_dataset.dataset)

    model = ClassificationModel(opt)

    correct = 0
    total = 0

    for step, data in enumerate(dataloader, 0):
        model.test(data)
        total += opt.batch_size
        correct += model.correct

    print('Accuracy of the network on %d test images: %d %%' % (
        dataset_size, 100 * correct / total
    ))

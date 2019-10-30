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

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, total_steps %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch+1)

    print('saving the last model at the end of epoch %d' % opt.n_epochs)
    model.save('latest')
    model.save(opt.n_epochs)

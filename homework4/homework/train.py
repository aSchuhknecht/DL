import torch
import numpy as np

from .models import Detector, save_model, ClassificationLoss
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = Detector().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """

    n_epochs = 1
    batch_size = 128
    loss = torch.nn.CrossEntropyLoss().to(device)

    trans = dense_transforms.Compose([
        dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor(),
        dense_transforms.ToHeatmap()
    ])

    # train_dataloader = load_dense_data('dense_data/train', num_workers=0, batch_size=32, transform=trans)
    train_dataloader = load_detection_data('dense_data/train', num_workers=0, batch_size=32, transform=trans)
    valid_dataloader = load_detection_data('dense_data/valid')


    #  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=30)

    global_step = 0
    acc = []
    iou = []

    for epoch in range(0, n_epochs):

        model.train()
        # confusionMatrix = ConfusionMatrix()
        for step, (data, labels, sz) in enumerate(train_dataloader):
            data, labels = data.to(device), labels.to(device)

            output = model(data)
            # print(output.size())
            # print(labels.size())

            # print(output.argmax(1))
            # print(labels)
            # print(o)

            # loss = ClassificationLoss()(output, labels.long())
            loss_val = loss(output, labels)

            optimizer.zero_grad()
            loss_val.backward()

            # confusionMatrix.add(output.argmax(1).detach().cpu(), labels.detach().cpu())

            # acc.append(accuracy(o, labels).detach().cpu().numpy())
            # acc.append(confusionMatrix.global_accuracy)
            # iou.append(confusionMatrix.iou)
            # print('loss: ', loss_val)

            train_logger.add_scalar('loss', float(1), global_step=global_step)

            optimizer.step()
            global_step += 1

        # train_logger.add_scalar('accuracy', np.mean(acc), global_step=global_step)
        # train_logger.add_scalar('iou', np.mean(iou), global_step=global_step)


        # print('train iou: ', np.mean(iou))
        # scheduler.step(np.mean(iou))

        # valid_data, valid_labels = next(iter(valid_dataloader))
        # valid_data, valid_labels = valid_data.to(device), valid_labels.to(device)

        # valid_pred = model(valid_data)
        # valid_acc = float(accuracy(valid_pred, valid_labels))
        # print('valid: ', valid_acc)

        # valid_logger.add_scalar('accuracy', valid_acc, global_step=global_step)

    # raise NotImplementedError('train')
    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

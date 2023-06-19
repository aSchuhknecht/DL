from .models import CNNClassifier, save_model, ClassificationLoss
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import numpy as np


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """

    n_epochs = 50
    batch_size = 128

    train_dataloader = load_data('data/train')
    valid_dataloader = load_data('data/valid')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    global_step = 0
    acc = []

    for epoch in range(0, n_epochs):
        for step, (data, labels) in enumerate(train_dataloader):

            o = model(data)
            # print(o.size())
            # print(o)

            loss = ClassificationLoss()(o, labels)
            acc.append(accuracy(o, labels))

            train_logger.add_scalar('loss', float(1), global_step=global_step)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            global_step += 1

        train_logger.add_scalar('accuracy', np.mean(acc), global_step=global_step)
        print(np.mean(acc))

        valid_data, valid_labels = next(iter(valid_dataloader))
        valid_pred = model(valid_data)
        valid_acc = accuracy(valid_pred, valid_labels)

        valid_logger.add_scalar('accuracy', valid_acc, global_step=global_step)

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

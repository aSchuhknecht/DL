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

    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """

    n_epochs = 30
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
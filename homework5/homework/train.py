from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms


def train(args):
    from os import path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = Planner().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """

    n_epochs = 20
    batch_size = 128

    trans = dense_transforms.Compose([
        dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor()
    ])

    train_dataloader = load_data('drive_data', num_workers=0, batch_size=128, transform=trans)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss = torch.nn.MSELoss()

    global_step = 0

    for epoch in range(0, n_epochs):

        print(epoch)
        model.train()
        # confusionMatrix = ConfusionMatrix()
        for data, labels in train_dataloader:
            data, labels = data.to(device), labels.to(device)

            output = model(data)
            # print(output)
            # print(labels)

            loss_val = loss(output, labels)

            optimizer.zero_grad()
            loss_val.backward()

            train_logger.add_scalar('loss', loss_val, global_step=global_step)
            # log(train_logger, data, labels, output, global_step)

            optimizer.step()
            global_step += 1

    save_model(model)


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

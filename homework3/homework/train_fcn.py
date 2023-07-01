import torch
import numpy as np

from .models import FCN, save_model, ClassificationLoss
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, accuracy
from . import dense_transforms
import torch.utils.tensorboard as tb
from torchvision import transforms


def train(args):
    from os import path

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = FCN().to(device)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """

    n_epochs = 10
    batch_size = 128

    trans = dense_transforms.Compose([
        dense_transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor()
    ])

    train_dataloader = load_dense_data('dense_data/train', num_workers=0, batch_size=32, transform=trans)
    valid_dataloader = load_dense_data('dense_data/valid')

    #  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    global_step = 0
    acc = []
    iou = []

    for epoch in range(0, n_epochs):
        for step, (data, labels) in enumerate(train_dataloader):
            data, labels = data.to(device), labels.to(device)

            output = model(data)
            # print(output.size())
            # print(labels.size())

            # print(output.argmax(1))
            # print(labels)
            # print(o)

            loss = ClassificationLoss()(output, labels.long())

            optimizer.zero_grad()
            #  loss.requires_grad = True
            loss.backward()

            confusionMatrix = ConfusionMatrix()
            confusionMatrix.add(output.argmax(1).detach().cpu(), labels.detach().cpu())

            # acc.append(accuracy(o, labels).detach().cpu().numpy())
            acc.append(confusionMatrix.global_accuracy)
            iou.append(confusionMatrix.iou)

            train_logger.add_scalar('loss', float(1), global_step=global_step)

            optimizer.step()
            global_step += 1

        train_logger.add_scalar('accuracy', np.mean(acc), global_step=global_step)
        train_logger.add_scalar('iou', np.mean(iou), global_step=global_step)

        print('train accuracy: ', np.mean(acc))
        print('train iou: ', np.mean(iou))

        valid_data, valid_labels = next(iter(valid_dataloader))
        valid_data, valid_labels = valid_data.to(device), valid_labels.to(device)

        valid_pred = model(valid_data)
        valid_acc = float(accuracy(valid_pred, valid_labels))
        print('valid: ', valid_acc)

        valid_logger.add_scalar('accuracy', valid_acc, global_step=global_step)

    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

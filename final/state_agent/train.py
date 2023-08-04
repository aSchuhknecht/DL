import torch
import torch.utils.tensorboard as tb
import numpy as np
from .util import load_data
from .model import Model

def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Model().to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        if args.run_number:
            args.log_dir += f'/{args.run_number}'
        print(f"Logging tensorboard data at {args.log_dir}")
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    train_data, valid_data = load_data('train_data', num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss = torch.nn.MSELoss()

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()

        for state, label in train_data:
            state, label = state.to(device), label.to(device)

            logit = model(state)
            loss_val = loss(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        model.eval()
        for state, label in valid_data:
            state, label = state.to(device), label.to(device)

            vlogit = model(state)
            vloss = loss(vlogit, label)

            if valid_logger:
                valid_logger.add_scalar('loss', vloss, global_step)

        print(f"Epoch: {epoch}")

        script = torch.jit.script(model)
        torch.jit.save(script, 'state_agent/matt_agent.pt')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-rn', '--run_number', type=int)

    args = parser.parse_args()
    train(args)
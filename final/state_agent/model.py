import torch


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)


class Model(torch.nn.Module):
    def __init__(self, layers=[16, 32, 64, 128, 256], n_input=11, n_output=3):
        super().__init__()

        net_layers = []

        c = n_input
        for i, layer in enumerate(layers):
            net_layers.append(torch.nn.Linear(c, layer))

            net_layers.append(torch.nn.ReLU())

            c = layer

        net_layers.append(torch.nn.Linear(c, c))
        net_layers.append(torch.nn.Linear(c, n_output))
        self.net = torch.nn.Sequential(*net_layers)

        self.net.apply(init_weights)
        self.skip = torch.nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.net(x) + self.skip(x)
import torch


class CNNClassifier(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Your code here
        """
        super().__init__(*args, **kwargs)

        c = 3
        lay = 32
        kern = 3
        layers = []

        layers.append(torch.nn.Conv2d(c, lay, kernel_size=(kern, kern)))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv2d(lay, 6, kernel_size=(1, 1)))

        self.network = torch.nn.Sequential(*layers)

        # raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """

        return self.network(x)
        # raise NotImplementedError('CNNClassifier.forward')


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r

import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))  # hi2

        Hint: Don't be too fancy, this is a one-liner
        """
        loss = torch.nn.CrossEntropyLoss()

        return loss(input, target)
        # raise NotImplementedError('ClassificationLoss.forward')


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=(3, 3), padding=1, stride=(stride, stride)),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_input, n_output, kernel_size=(3, 3), padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.downsample = None
            if n_input != n_output or stride != 1:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, kernel_size=(1, 1), stride=(stride, stride)),
                                                      torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            # return self.net(x) + identity
            return self.net(x)

    def __init__(self, layers=None, n_input_channels=3):
        """
        Your code here
        """
        super().__init__()
        if layers is None:
            layers = [64, 64, 64]

        # c = 3
        # lay = 32
        # kern = 3
        # layers = []
        #
        # layers.append(torch.nn.Conv2d(c, lay, kernel_size=(kern, kern)))
        # layers.append(torch.nn.ReLU())
        # layers.append(torch.nn.Conv2d(lay, lay, kernel_size=(1, 1)))

        L = [
            torch.nn.Conv2d(n_input_channels, 64, kernel_size=(7, 7), padding=3, stride=(2, 2)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        c = 64
        for lay in layers:
            L.append(self.Block(c, lay, stride=2))
            c = lay

        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 6)

        # raise NotImplementedError('CNNClassifier.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """

        z = self.network(x)
        z = z.mean([2, 3])
        return self.classifier(z)

        # raise NotImplementedError('CNNClassifier.forward')


class FCN(torch.nn.Module):

    class DownBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=(7, 7), padding=3, stride=(stride, stride)),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.downsample = None
            if n_input != n_output or stride != 1:
                self.downsample = torch.nn.Sequential(torch.nn.Conv2d(n_input, n_output, kernel_size=(1, 1), stride=(stride, stride)),
                                                      torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.net(x) + identity
            # return self.net(x)

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=(7, 7), padding=(3, 3), stride =(stride, stride), output_padding=(1,1)),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            self.downsample = None
            if n_input != n_output or stride != 1:
                self.downsample = torch.nn.Sequential(torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=(1, 1), stride=(stride, stride), output_padding=(1,1)),
                                                      torch.nn.BatchNorm2d(n_output))

        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(identity)
            return self.net(x) + identity
            # return self.net(x)

    def __init__(self, layers=None, n_input_channels=3):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """
        if layers is None:
            layers = [32, 64, 128]

        down_layers = [32, 64, 128]
        up_layers = [64, 32, 5]

        L = []
        c = 3
        for lay in down_layers:
            L.append(self.DownBlock(c, lay, stride=2))
            c = lay
        for lay in up_layers:
            L.append(self.UpBlock(c, lay, stride=2))
            c = lay

        self.network = torch.nn.Sequential(*L)

        self.downBlock1 = self.DownBlock(3, 32, stride=2)
        self.downBlock2 = self.DownBlock(32, 64, stride=2)
        self.downBlock3 = self.DownBlock(64, 128, stride=2)

        self.UpBlock1 = self.UpBlock(128, 64, stride=2)
        self.UpBlock2 = self.UpBlock(64*2, 32, stride=2)
        self.UpBlock3 = self.UpBlock(32, 5, stride=2)
        # self.classifier = torch.nn.Linear(c, 6)

        # raise NotImplementedError('FCN.__init__')

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """
        # z = self.network(x)

        d1_in = x.size()
        d1 = self.downBlock1(x)  # output of  first downConv

        d2_in = d1.size()
        d2 = self.downBlock2(d1)  # output of 2nd downConv

        d3_in = d2.size()
        d3 = self.downBlock3(d2)  # output of 3rd downConv

        u1 = self.UpBlock1(d3)
        u1 = u1[:, :, :d3_in[2], :d3_in[3]]
        u1 = torch.cat((u1, d2), dim=1)  # skip connection

        u2 = self.UpBlock2(u1)
        u2 = u2[:, :, :d2_in[2], :d1_in[3]]
        # u2 = torch.cat((u2, d1), dim=1)  # skip connection

        u3 = self.UpBlock3(u2)
        u3 = u3[:, :, :d1_in[2], :d1_in[3]]

        return u3
        # raise NotImplementedError('FCN.forward')


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r

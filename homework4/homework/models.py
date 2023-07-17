import torch
import torch.nn.functional as F
import numpy as np
import math


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


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pad = math.floor(max_pool_ks/2)

    hm = heatmap[None, None]
    hm = hm.to(device)

    out = F.max_pool2d(hm, max_pool_ks, padding=(pad, pad), stride=1)
    out = out.to(device)

    out = torch.squeeze(out)

    hm_flat = heatmap.flatten()
    out_flat = out.flatten()

    hm_flat = hm_flat.to(device)
    out_flat = out_flat.to(device)

    # num_max = 0
    # for i in range(0, hm_flat.size(0)):
    #     if hm_flat[i] != out_flat[i] or out_flat[i] <= min_score:
    #         out_flat[i] = float('-inf')
    #         x = 2
    #     else:
    #         num_max = num_max + 1
    #
    # if num_max > max_det:
    #     num_max = max_det
    # print(num_max)
    #
    # values0, indices0 = torch.topk(out_flat, num_max)
    # print(indices0)

    mask = torch.eq(hm, out)
    # print(hm)
    # print(out)
    # print(mask)

    res = torch.masked_select(hm, mask)
    res = torch.where(res > min_score, res, 0)
    res2 = hm.masked_fill(mask == 0, 0.0)
    # res2 = torch.where(res2 > min_score, res2, 0)

    num_max = torch.count_nonzero(res).cpu().numpy().item()
    if num_max > max_det:
        num_max = max_det
    # print(num_max)

    values1, indices1 = torch.topk(res2.flatten(), num_max)
    # print(indices1)

    num_cols = heatmap.size(1)

    scores = []
    for i in range(0, num_max):
        cx = indices1[i] % num_cols
        cy = indices1[i] // num_cols

        tup = (values1[i].detach().cpu().numpy().item(), cx.detach().cpu().numpy().item(), cy.detach().cpu().numpy().item())
        # tup = (values1[i], cx, cy)
        scores.append(tup)

    # print(scores)
    return scores
    # raise NotImplementedError('extract_peak')


class Detector(torch.nn.Module):

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
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()

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
        self.UpBlock2 = self.UpBlock(64 * 2, 32, stride=2)
        self.UpBlock3 = self.UpBlock(32 * 2, 3, stride=2)

        # raise NotImplementedError('Detector.__init__')

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """

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
        u2 = u2[:, :, :d2_in[2], :d2_in[3]]
        u2 = torch.cat((u2, d1), dim=1)  # skip connection

        u3 = self.UpBlock3(u2)
        u3 = u3[:, :, :d1_in[2], :d1_in[3]]

        return u3
        # return torch.nn.Sigmoid(u3)
        # raise NotImplementedError('Detector.forward')

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        self.eval()

        image = image[None]
        result = self.forward(image)
        result = result.squeeze()
        # print("here")

        t1 = extract_peak(result[0, :, :], max_det=30)
        t2 = extract_peak(result[1, :, :], max_det=30)
        t3 = extract_peak(result[2, :, :], max_det=30)

        r1, r2, r3 = [], [], []
        for i in range(0, len(t1)):
            tup = t1[i] + (0.0, 0.0)
            r1. append(tup)
        for i in range(0, len(t2)):
            tup = t2[i] + (0.0, 0.0)
            r2.append(tup)
        for i in range(0, len(t3)):
            tup = t3[i] + (0.0, 0.0)
            r3.append(tup)

        return r1, r2, r3

        # raise NotImplementedError('Detector.detect')


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()

import torch
import torch.nn.functional as F
import numpy as np
import math
from . import dense_transforms


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

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride, bias=False)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 128], n_output_channels=3, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        # self.classifier = torch.nn.Conv2d(c, n_output_channels, 1)
        self.classifier = torch.nn.Sequential(torch.nn.Conv2d(c, n_output_channels, 1),
                                           torch.nn.Sigmoid())

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d' % i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d' % i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        return self.classifier(z)

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

        print(r2)
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

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg16


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Perceptual_Loss(nn.Module):
    def __init__(self, p_list):
        super().__init__()
        self.net = vgg16(pretrained=True).features.eval()
        self.p_list = p_list
        self.features = []
        for i in p_list:
            # print(self.net[i])
            self.net[i].register_forward_hook(self.forward_hook_fn)

    def forward(self, y_pre, y):
        self.net(torch.cat([y_pre, y], dim=0))
        return self.get_loss()

    def forward_hook_fn(self, model, inputs, outputs):
        # print(model)
        self.features.append(outputs.clone().data)

    def get_loss(self):
        loss_total = 0
        for i in range(len(self.p_list)):
            b = self.features[0].shape[0]
            y_pre_feature = self.features[i][:b // 2]
            y_feature = self.features[i][b // 2:]
            loss = torch.pow(y_pre_feature - y_feature, 2).sum() ** 0.5
            c, h, w = y_feature.size(1), y_feature.size(2), y_feature.size(3)
            loss_total += loss.data / (c * h * w)
        self.features = []
        return loss_total


def gauss(kernel_size, sigma):
    kernel = torch.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = math.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel


class Color_Loss(nn.Module):
    def __init__(self, kernel_size=None, sigma=None, use_gauss=False):
        super().__init__()
        self.use_gauss = use_gauss
        if use_gauss:
            if kernel_size is None or sigma is None:
                raise Exception("kernel size or sigma 不能为空")
            self.kernel = gauss(kernel_size, sigma).unsqueeze(0).unsqueeze(0)
            self.kernel = self.kernel.expand((3, 3, kernel_size, kernel_size))

    def forward(self, y_pre, y):
        # kernel size = (out_channels，in_channe/groups，H，W)
        if self.use_gauss:
            kernel = self.kernel.to(y.device)
            y_pre = nn.functional.conv2d(y_pre, weight=kernel, stride=1)
            y = nn.functional.conv2d(y, weight=kernel, stride=1)
        return torch.cosine_similarity(y_pre.float(), y.float()).mean()


def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=1023, size_average=True, full=False):
    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False, weights=None):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3, weights=None):
        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range,
                       weights=self.weights)


if __name__ == '__main__':
    y_pre = torch.randn((1, 3, 448, 448)).cuda()
    y = torch.randn((1, 3, 448, 448)).cuda()
    loss = Color_Loss(21, 5).cuda()
    color_loss = loss(y_pre, y)
    print(color_loss)
    p = Perceptual_Loss([])
    print(p.net)

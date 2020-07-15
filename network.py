import torch
from model.modules import conv, RRG, nonlinearity, Dblock, DecoderBlock, Upsampling
from torch import nn
from torchvision import models


class Raw2Rgb(nn.Module):
    def __init__(self, input_nc=4, conv=conv):
        super(Raw2Rgb, self).__init__()
        output_nc = 3

        num_rrg = 3
        num_dab = 5
        n_feats = 96
        kernel_size = 3
        reduction = 8
        self.input_nc = input_nc
        act = nn.ReLU(True)

        modules_head = [conv(input_nc, n_feats, kernel_size=kernel_size, stride=1)]

        modules_body = [
            RRG(
                conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab) \
            for _ in range(num_rrg)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_body.append(act)

        modules_tail = [conv(n_feats, n_feats, kernel_size), act]
        modules_tail_rgb = [conv(n_feats, output_nc * 4, kernel_size=1)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.tail_rgb = nn.Sequential(*modules_tail_rgb)
        if self.input_nc==4:
            self.output = conv(output_nc, output_nc, kernel_size=3)
        else:
            self.output = conv(output_nc*4, output_nc, kernel_size=3)

    def forward(self, x):
        x = self.head(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
        x = self.body[-1](x)
        x = self.tail(x)
        x = self.tail_rgb(x)
        if self.input_nc==4:
            x = nn.functional.pixel_shuffle(x, 2)
        x = self.output(x)
        return (torch.tanh_(x) + 1) / 2


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    model = Raw2Rgb(input_nc=4).cuda()
    # model = Discriminator().cuda()
    # model.apply(weights_init)
    # print()
    raw = torch.randn((1, 4, 224, 224)).cuda()
    # raw = torch.randn((1, 3, 448, 448)).cuda()
    output = model(raw)
    print(output.shape)
    # print(model)

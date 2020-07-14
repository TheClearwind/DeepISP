import torch
from torch import nn
from torchvision import models

from model.modules import conv, RRG, nonlinearity, Dblock, DecoderBlock, Upsampling


class Raw2Rgb(nn.Module):
    def __init__(self, input_nc=4, conv=conv):
        super(Raw2Rgb, self).__init__()
        output_nc = 3

        num_rrg = 3
        num_dab = 5
        n_feats = 96
        kernel_size = 3
        reduction = 8

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
        self.output = conv(output_nc, output_nc, kernel_size=3)
        # conv1x1 = [conv(n_feats * 2, n_feats, kernel_size=1)]
        # self.conv1x1 = nn.Sequential(*conv1x1)

    def forward(self, x):
        x = self.head(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
        x = self.body[-1](x)
        x = self.tail(x)
        x = self.tail_rgb(x)
        x = nn.functional.pixel_shuffle(x, 2)
        x = self.output(x)
        return (torch.tanh_(x) + 1) / 2


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()
        self.conv_1_3 = nn.Conv2d(1, 3, kernel_size=1)
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        gan_feature = self.main(x)
        out_src = self.conv1(gan_feature)
        return out_src


class FuseNet(nn.Module):
    def __init__(self, model_paths):
        super().__init__()
        self.model_paths = model_paths
        self.net = Raw2Rgb()
        self.net.eval()
        self.conv = nn.Sequential(
            nn.Conv2d(30, 64, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.output = nn.Conv2d(128, 3, kernel_size=1)

    def forward(self, inputs):
        resuls = []
        for each in self.model_paths:
            with torch.no_grad():
                self.net.load_state_dict(torch.load(each))
                resuls.append(self.net(inputs))
        img_inputs = torch.cat(resuls, dim=1)
        img_hat = self.conv(img_inputs)
        img_hat = self.output(img_hat)
        return (torch.tanh(img_hat) + 1) / 2
        # return torch.sigmoid(img_hat)


class DinkNet34(nn.Module):
    def __init__(self, out_ch=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = Upsampling(filters[0], 32, 2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_ch, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return (torch.tanh(out) + 1) * 0.5


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

import argparse
import datetime
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageData
from logger import Logger
from loss import MS_SSIM, Color_Loss, Perceptual_Loss, TVLoss
from network import weights_init, Raw2Rgb
from unet_plus import NestedUNet

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

torch.manual_seed(2020)  # cpu
torch.cuda.manual_seed(2020)  # gpu

parse = argparse.ArgumentParser()
parse.add_argument("-train_data", type=str, default="./")
parse.add_argument("-batch_size", type=int, default=1)
parse.add_argument("-data_size", type=int, default=5000)
parse.add_argument("-epochs", type=int, default=60)
parse.add_argument("-G_step", type=int, default=5)
parse.add_argument('-lr', type=float, default=1e-4, help='learning rate, default=0.0001')
parse.add_argument("-beta1", type=float, default=0.9)
parse.add_argument("-beta2", type=float, default=0.999)
parse.add_argument("-use_tb", type=bool, default=False)
parse.add_argument("-log_step", type=int, default=30)
parse.add_argument("-log_dir", type=str, default="./logs/{}".format(datetime.datetime.now().strftime('%Y-%m-%d-%S')))
parse.add_argument("-save_path", type=str, default="./save_model")
parse.add_argument("-recover", type=bool, default=False)
parse.add_argument("-weight_path", type=str, default=None)
config = parse.parse_args()
if not os.path.exists(config.save_path):
    os.mkdir(config.save_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
if config.use_tb:
    logger = Logger(config.log_dir)
else:
    logger = None
# 定义数据集
transform = transforms.Compose([
    transforms.ToTensor()
])
train_data = ImageData(config.train_data, data_size=config.data_size, transform=transform)
# 定义损失函数
rec_loss_fn = nn.L1Loss().to(device)
# rec_loss_fn = nn.MSELoss().to(device)
color_loss_fn = Color_Loss().to(device)
perceptual_loss_fn = Perceptual_Loss([28]).to(device)
# ssim_fn = MS_SSIM().to(device)
tv_loss_fn = TVLoss().to(device)


def criterion(y_hat, y):
    # 重建损失
    rec_loss = rec_loss_fn(y_hat, y)
    # # 颜色损失
    color_loss = 0.5 * (1 - color_loss_fn(y_hat, y)) ** 2
    # SSIM
    # print(img_hat.dtype)
    # print(img.dtype)
    # ssim_loss = 1 - ssim_fn(y_hat, y)
    # 感知损失
    perceptual_loss = perceptual_loss_fn(y_hat, y)
    # tv loss
    tv_loss = tv_loss_fn(y_hat)

    loss_total = 5 * rec_loss + 0.1 * color_loss + 0.01 * tv_loss + 1 * perceptual_loss

    return [loss_total, rec_loss, color_loss, perceptual_loss, tv_loss]


dataloader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)
# 定义模型
raw2rgb = Raw2Rgb(input_nc=4).to(device)
# raw2rgb = DinkNet34(out_ch=3).to(device)
# raw2rgb = NestedUNet(out_ch=3, input_channels=3, deep_supervision=True).to(device)
raw2rgb.apply(weights_init)
raw2rgb.train()
if config.recover:
    print("recover model from ", config.weight_path)
    raw2rgb.load_state_dict(torch.load(config.weight_path))
    start = int(os.path.splitext(os.path.basename(config.weight_path))[0].split("_")[-1])
else:
    start = 1
# 定义变量
iteration = len(train_data) // config.batch_size
num_iter = 0
# 定义优化器
optimizer_G = optim.Adam(raw2rgb.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

raw2rgb, optimizer_G = amp.initialize(raw2rgb, optimizer_G, opt_level="O1")
for epoch in range(start, config.epochs + 1):
    for index, (raw, img) in enumerate(dataloader):
        raw = raw.to(device)
        img = img.to(device).half()
        loss = {}
        img_hat = raw2rgb(raw)
        loss_g_total = 0
        rec_loss_total = 0
        color_loss_total = 0
        perceptual_loss_total = 0
        tv_loss_total = 0

        loss_g, rec_loss, color_loss, perceptual_loss, tv_loss = criterion(img_hat, img)
        loss_g_total += loss_g
        rec_loss_total += rec_loss
        color_loss_total += color_loss
        perceptual_loss_total += perceptual_loss
        tv_loss_total += tv_loss
        optimizer_G.zero_grad()

        with amp.scale_loss(loss_g_total, optimizer_G) as scaled_g_loss:
            scaled_g_loss.backward()
        optimizer_G.step()

        loss["model/rec_loss"] = rec_loss_total.item()
        loss["model/color_loss"] = color_loss_total.item()
        loss["model/perceptual_loss"] = perceptual_loss_total.item()
        loss["model/tv_loss"] = tv_loss_total.item()

        log = "epoch:{}/{} iteration:{}/{}".format(epoch, config.epochs, index, iteration)
        if index % config.log_step == 0:
            for tag, value in loss.items():
                log += ", {}: {:.6f}".format(tag, value)
            print(log)
            if logger is not None:
                for tag, value in loss.items():
                    logger.scalar_summary(tag, value, num_iter + 1)
                num_iter += 1
    if epoch % 5 == 0:
        torch.save(raw2rgb.state_dict(), os.path.join(config.save_path, "raw2rgb_%d.pth" % epoch))

torch.save(raw2rgb.state_dict(), os.path.join(config.save_path, "raw2rgb.pth"))

import os

import imageio
import numpy as np
import torch
import torchvision
from torchvision import transforms

from network import Raw2Rgb
from utils import pack_raw
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("-weights",type=str,default="")
parse.add_argument("-dataset",type=str,default="AIM2020_ISP_validation_raw")
config = parse.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.ToTensor()
model = Raw2Rgb(input_nc=4).to(device)
model.eval()
model.load_state_dict(torch.load(config.weights))
model.half()
raw_path = config.dataset
for path in os.listdir(raw_path):
    raw_image = np.asarray(imageio.imread(os.path.join(raw_path, path)))
    raw_image = pack_raw(raw_image).astype(np.float32) / (4 * 255)
    raw_image = torch.tensor(raw_image)
    raw_image = raw_image.unsqueeze(0).to(device).half()

    with torch.no_grad():
        img_hat = model(raw_image)
        print("process %s " % path, img_hat.shape)
        torchvision.utils.save_image(img_hat, "raw2rgb/valtest/" + os.path.basename(path))
    print("Done!")

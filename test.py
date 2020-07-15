import os
import time
import imageio
import numpy as np
import torch
import torchvision
from torchvision import transforms

from network import Raw2Rgb
from utils import pack_raw,pack_raw_v2
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("-weights",type=str,default="")
parse.add_argument("-in_nc",type=int,default=3)
parse.add_argument("-dataset",type=str,default="AIM2020_ISP_validation_raw")
config = parse.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.ToTensor()
model = Raw2Rgb(input_nc=config.in_nc).to(device)
model.eval()
model.load_state_dict(torch.load(config.weights))
model.half()
raw_path = config.dataset
total_time = 0
count = 0
for path in os.listdir(raw_path):
    raw_image = np.asarray(imageio.imread(os.path.join(raw_path, path)))
    if config.in_nc == 4:
        raw_image = pack_raw(raw_image).astype(np.float32) / (4 * 255)
    else:
        raw_image = pack_raw_v2(raw_image).astype(np.float32) / (4 * 255)
    raw_image = torch.tensor(raw_image)
    raw_image = raw_image.unsqueeze(0).to(device).half()

    with torch.no_grad():
        start = time.time()
        img_hat = model(raw_image)
        torch.cuda.synchronize()
        stamp = time.time() - start
        print("process %s time:%f" % (path,stamp), img_hat.shape)
        total_time+=stamp
        count+=1
        torchvision.utils.save_image(img_hat, "raw2rgb/test/" + os.path.basename(path))
print("Done! avg time:%f"%(total_time/count))

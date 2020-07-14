import os
import random
from glob import glob

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import pack_raw_v2, pack_raw


class ImageData(Dataset):

    def __init__(self, dataset_dir, data_size=-1, mode="train", transform=None):
        self.raw_dir = glob(os.path.join(dataset_dir, mode, 'huawei_raw', "*.png"))
        self.dslr_dir = glob(os.path.join(dataset_dir, mode, 'canon', "*.jpg"))

        self.dataset_size = data_size
        self.transform = transform

        self.raw_dir = sorted(self.raw_dir, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.dslr_dir = sorted(self.dslr_dir, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def __len__(self):
        if self.dataset_size == -1:
            return len(self.raw_dir)
        return self.dataset_size

    def __getitem__(self, idx):
        raw_path = self.raw_dir[idx]
        raw_image = np.asarray(imageio.imread(raw_path))
        raw_image = pack_raw(raw_image).astype(np.float32) / (4 * 255)
        # raw_image = pack_raw_v2(raw_image).astype(np.float32) / (4 * 255)
        raw_image = torch.tensor(raw_image)
        img_path = self.dslr_dir[idx]
        dslr_image = imageio.imread(img_path)
        return raw_image, self.transform(dslr_image)
        # return raw_image,dslr_image


if __name__ == '__main__':
    dataset = ImageData("./", mode="test", transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))
    idx = random.randint(0, len(dataset) - 1)
    r, i = dataset[idx]
    print(r.shape)
    print(r.max())
    print(r.min())
    # print(i.shape)

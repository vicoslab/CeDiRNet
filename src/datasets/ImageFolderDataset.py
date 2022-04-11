import glob
import os

import numpy as np
from PIL import Image, ImageFile

from torch.utils.data import Dataset

import torch
from torchvision.transforms import functional as F
from utils.transforms import Padding, ToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageFolderDataset(Dataset):

    def __init__(self, root_dir='./', pattern='*.jpg', resize_factor=None, normalize=False):

        print('ImageFolderDataset created')

        self.resize_factor = resize_factor
        self.normalize = normalize

        # get image and instance list
        image_list = glob.glob(os.path.join(root_dir, pattern))
        image_list.sort()

        self.image_list = image_list
        self.real_size = len(self.image_list)

        print('found %d images' % self.real_size)

        self.pad = Padding(keys=('image',), pad_to_size_factor=32)
        self.to_tensor = ToTensor(keys=('image',), type=(torch.FloatTensor,))

    def __len__(self):
        return self.real_size

    def __getitem__(self, index):
        # this will load only image info but not the whole data
        image = Image.open(self.image_list[index])
        im_size = image.size

        if self.resize_factor is not None:
            im_size = int(image.size[0] * self.resize_factor), int(image.size[1] * self.resize_factor)

        if self.resize_factor is not None and self.resize_factor != 1.0:
            image = image.resize(im_size, Image.BILINEAR)

        sample = dict(image=image,
                      im_name=self.image_list[index],
                      im_size=im_size,
                      index=index)

        sample = self.pad(sample)
        sample = self.to_tensor(sample)

        if self.normalize:
            sample['image'] = (np.array(sample['image']) - 128)/ 128  # normalize to [-1,1]

        return sample

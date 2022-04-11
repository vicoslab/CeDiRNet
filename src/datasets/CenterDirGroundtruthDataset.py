import os.path
import numpy as np

import torch
from torch.utils.data import Dataset
from datasets.LockableSeedRandomAccess import LockableSeedRandomAccess

class CenterDirGroundtruthDataset(Dataset, LockableSeedRandomAccess):

    def __init__(self, dataset, centerdir_groundtruth_op):

        self.dataset = dataset
        self.centerdir_groundtruth_op = centerdir_groundtruth_op

    def lock_samples_seed(self, index_list):
        if isinstance(self.dataset,LockableSeedRandomAccess):
            self.dataset.lock_samples_seed(index_list)
        #else:
        #    print("Warning: underlying dataset not instance of LockableSeedRandomAccess .. skipping")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        image = sample['image']

        sample['centerdir_groundtruth'] = self.centerdir_groundtruth_op._create_empty(image.shape[-2], image.shape[-1])

        return sample
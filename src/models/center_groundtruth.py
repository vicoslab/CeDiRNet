import os
import numpy as np

import torch
import torch.nn as nn

from utils.utils import GaussianLayer

class CenterDirGroundtruth(nn.Module):
    def __init__(self, use_cached_backbone_output=False, add_synthetic_output=False,
                 center_extend_px=3, center_gt_blur=2):
        super().__init__()

        self.use_cached_backbone_output = use_cached_backbone_output

        self.add_synthetic_output = add_synthetic_output

        self.center_extend_px = center_extend_px

        ################################################################
        # Prepare buffer for location/coordinate map
        xym = self._create_xym(0)

        self.register_buffer("xym", xym, persistent=False)

        with torch.no_grad():
            self.gaussian_blur = GaussianLayer(num_channels=1, sigma=center_gt_blur)

    def _create_xym(self, size):
        # coordinate map
        # CAUTION: original code may not have correctly aligned offsets
        #          since xm[-1] will not be MAX_OFFSET-1, but MAX_OFFSET
        #          -> this has been fixed by adding +1 element to xm and ym array
        align_fix = 1

        xm = torch.linspace(0, 1, size + align_fix).view(1, 1, -1).expand(1, size + align_fix, size + align_fix) * size
        ym = torch.linspace(0, 1, size + align_fix).view(1, -1, 1).expand(1, size + align_fix, size + align_fix) * size
        return torch.cat((xm, ym), 0)

    def _get_xym(self, height, width):
        max_size = max(height, width)
        if max_size > min(self.xym.shape[1], self.xym.shape[2]):
            self.xym = self._create_xym(max_size).to(self.xym.device)

        return self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

    def forward(self, sample, batch_index):
        image = sample['image']
        label = sample['label']

        centers = sample.get('center')
        centerdir_gt = sample.get('centerdir_groundtruth')
        output = sample.get('output')

        # generate any missing groundtruth
        centerdir_gt = self._generate_groundtruth(centerdir_gt, image.shape, centers)

        # also create synthetic output if does not exist yet
        if self.add_synthetic_output:
            if output is None:
                output = self._create_synthetic_output(centerdir_gt, label)

            sample['output'] = output

        # attach center direction groundtruh and output values to returned sample
        sample['centerdir_groundtruth'] = centerdir_gt

        return sample

    def _generate_groundtruth(self, centerdir_gt, input_shape, centers):
        # requires list of centers
        assert centers is not None

        with torch.no_grad():
            batch_size, _, height, width = input_shape

            xym_s = self._get_xym(height, width)

            # center direction losses
            for b in range(batch_size):
                # requires prepared centerdir_gt
                assert centerdir_gt is not None and len(centerdir_gt[0][b]) >= 5

                gt_center_mask = torch.ones([1, height, width], dtype=torch.uint8, device=xym_s.device)

                # list of all centers valid for this batch
                valid_centers = (centers[b,:,0] > 0) | (centers[b,:,1] > 0)

                # skip if no centers
                if valid_centers.sum() > 0:
                    assigned_center_ids = self._find_closest_center(centers[b], height, width, xym_s.device)

                    # per-pixel center locations
                    gt_center_x = centers[b,assigned_center_ids[:], 1].unsqueeze(0)
                    gt_center_y = centers[b,assigned_center_ids[:], 0].unsqueeze(0)

                    gt_X = gt_center_x - xym_s[1].unsqueeze(0)
                    gt_Y = gt_center_y - xym_s[0].unsqueeze(0)

                    gt_center_mask *= ~((gt_X.abs() < self.center_extend_px) * (gt_Y.abs() < self.center_extend_px))

                    gt_R = torch.sqrt(torch.pow(gt_X, 2) + torch.pow(gt_Y, 2))
                    gt_theta = torch.atan2(gt_Y, gt_X)
                    gt_sin_th = torch.sin(gt_theta)
                    gt_cos_th = torch.cos(gt_theta)

                    # normalize groundtruth vector to 1 for all instance pixels
                    gt_M = torch.sqrt(torch.pow(gt_sin_th, 2) + torch.pow(gt_cos_th, 2))
                    gt_sin_th = gt_sin_th / gt_M
                    gt_cos_th = gt_cos_th / gt_M

                    centerdir_gt[0][b, 0] = gt_R
                    centerdir_gt[0][b, 1] = gt_theta
                    centerdir_gt[0][b, 2] = gt_sin_th
                    centerdir_gt[0][b, 3] = gt_cos_th

                if gt_center_mask.all():
                    gt_center_mask = torch.zeros_like(gt_center_mask)
                else:
                    gt_center_mask = self.gaussian_blur(1 - gt_center_mask.unsqueeze(0).float())[0]
                    gt_center_mask /= gt_center_mask.max()

                centerdir_gt[0][b, 5] = gt_center_mask

        return centerdir_gt

    @staticmethod
    def parse_groundtruth(centerdir_gt):
        gt_R, gt_theta, gt_sin_th, gt_cos_th = centerdir_gt[0][:, 0], centerdir_gt[0][:, 1], centerdir_gt[0][:, 2], centerdir_gt[0][:,3]

        gt_center_mask = centerdir_gt[0][:, 5]

        return gt_R, gt_theta, gt_sin_th, gt_cos_th, gt_center_mask

    @staticmethod
    def convert_gt_centers_to_dictionary(gt_centers, instances, ignore=None):

        gt_centers_dict = [{id: gt_centers[b, id, :2].cpu().numpy()
                            for id in range(gt_centers[b].shape[0]) if gt_centers[b, id, 0] > 0 and gt_centers[b, id, 1] > 0 and id in torch.unique(instances[b])}
                                for b in range(len(gt_centers))]
        if ignore is not None:
            gt_centers_dict = [{k: c for k, c in gt_centers_dict[b].items() if ignore[b, 0][instances[b] == k].min() == 0}
                                    for b in range(len(gt_centers))]

        return gt_centers_dict

    def _create_empty(self, h, w):
        centerdir_gt_matrix = torch.zeros(size=[6, 1, h, w], dtype=torch.float, requires_grad=False)

        return (centerdir_gt_matrix,)

    def _create_synthetic_output(self, centerdir_gt, label):
        output = torch.cat((centerdir_gt[0][:,2],
                            centerdir_gt[0][:,3],
                            torch.log10(centerdir_gt[0][:,0] * self.MAX_OFFSET + 1),
                            label.float()),dim=1)
        return output

    def insert_cached_model(self, model, sample):

        if not self.use_cached_backbone_output:
            return model
        else:
            # save actual model (model.module) to prevent recursive construction (and consequential GPU memory leaks)
            return CachedOutputModel(model.module, sample['output'], self.use_cached_backbone_output)

    def _find_closest_center(self, centers, height, width, device):
        # get X and Y coordinates (in normalized range of 0..1)
        xym_s = self._get_xym(height, width)
        X, Y = xym_s[1], xym_s[0]

        # function used to calc closest distance
        def _calc_closest_center_patch_i(_center_x,_center_y, _X, _Y):
            # distance in cartesian space to center
            distances_to_center = torch.sqrt((_X[...,None] - _center_x) ** 2 + (_Y[..., None] - _center_y) ** 2)

            closest_center_index = torch.argmin(distances_to_center, dim=-1)

            return closest_center_index.long()

        # select patch size that is dividable but still as large as possible (from ranges of 16 to 128 - from 2**4 to 2**7)
        patch_size_options = [(2**i)*(2**j) for j in range(4,8) for i in range(4,8)]
        patch_size_options = [s for s in patch_size_options if height*width % s == 0]
        patch_size = max(patch_size_options)
        patch_count = (height*width) // patch_size

        # reshape all needed matrices into new shape
        (_X, _Y) = [x.reshape(patch_count,-1)  for x in [X, Y]]

        # main section to calc closest dist by splitting it
        valid_centers = (centers[:, 0] > 0) | (centers[:, 1] > 0) # use only valid centers and then remap indexes

        center_y = centers[valid_centers, 0]
        center_x = centers[valid_centers, 1]

        closest_center_index = torch.zeros((patch_count, patch_size), dtype=torch.long, device=device)
        for i in range(patch_count):
            closest_center_index[i, :] = _calc_closest_center_patch_i(center_x, center_y, _X[i], _Y[i])

        # re-map indexes from list of selected/valid center to list of all centers
        valid_centers_idx = torch.nonzero(valid_centers)
        closest_center_index = valid_centers_idx[closest_center_index]

        return closest_center_index.reshape([height, width])

class CachedOutputModel:
    def __init__(self, module, output, use_cached_backbone_output):
        self.module = module
        self.output = output
        self.use_cached_backbone_output = use_cached_backbone_output

    def __call__(self, input):
        if self.use_cached_backbone_output:
            output = self.output.to(input.device)
        else:
            output = self.module(input)

        return output

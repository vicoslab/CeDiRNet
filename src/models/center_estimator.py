import torch
import torch.nn as nn

import numpy as np

from models.center_augmentator import CenterDirAugmentator

from models.localization_networks import Conv1dMultiscaleLocalization, Conv2dDilatedLocalization

class CenterDirEstimator(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.return_backbone_only = False
        instance_center_estimator_op = Conv1dMultiscaleLocalization
        if args.get('use_learnable_nn'):
            from functools import partial
            instance_center_estimator_op = partial(Conv2dDilatedLocalization,
                                                   **args.get('learnable_nn_args',{}))

        self.instance_center_estimator = instance_center_estimator_op(
            local_max_thr=args.get('local_max_thr', 0.1),
            apply_input_smoothing_for_local_max=args.get('apply_input_smoothing_for_local_max', True))

        if args.get('augmentation'):
            self.center_augmentator = CenterDirAugmentator(
                **args.get('augmentation_kwargs'),
            )
        else:
            self.center_augmentator = None

        self.MAX_NUM_CENTERS = 4*512

    def init_output(self):
        self.instance_center_estimator.init_output()

        return input

    def forward(self, input, **gt):
        if self.center_augmentator is not None:
            input = self.center_augmentator(input, **gt)

        prediction_sin = input[:, 0].unsqueeze(1)
        prediction_cos = input[:, 1].unsqueeze(1)

        ignore = None
        if not self.training and 'ignore' in gt:
            # consider all ignore flags except DIFFICULT and padding (8==DIFFICULT; 64,128=PADDING) one which will be handled by evaluation
            ignore = gt['ignore'] & (255-8-64-128)

        center_pred, conv_resp = self.instance_center_estimator(prediction_cos, prediction_sin, ignore_region=ignore)

        center_pred = torch.tensor(center_pred).to(input.device)
        conv_resp = conv_resp[:,0]

        # convert center prediction list to tensor of fixed size so that it can be merged from parallel GPU processings
        center_pred = self._pack_center_predictions(center_pred, batch_size=len(input))

        return dict(output=input, center_pred=center_pred, center_heatmap=conv_resp)

    def _pack_center_predictions(self, center_pred, batch_size):
        center_pred_all = torch.zeros((batch_size, self.MAX_NUM_CENTERS, center_pred.shape[1] if len(center_pred) > 0 else 5),
                                      dtype=torch.float, device=center_pred.device)
        if len(center_pred) > 0:
            for b in center_pred[:, 0].unique().long():
                valid_centers_idx = torch.nonzero(center_pred[:, 0] == b.float()).squeeze(dim=1).long()

                if len(valid_centers_idx) > self.MAX_NUM_CENTERS:
                    valid_centers_idx = valid_centers_idx[:self.MAX_NUM_CENTERS]
                    print('WARNING: got more centers (%d) than allowed (%d) - removing last centers to meed criteria' % (len(valid_centers_idx), self.MAX_NUM_CENTERS))

                center_pred_all[b, :len(valid_centers_idx), 0] = 1
                center_pred_all[b, :len(valid_centers_idx), 1:] = center_pred[valid_centers_idx, 1:]

        return center_pred_all

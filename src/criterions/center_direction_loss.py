import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from models.center_groundtruth import CenterDirGroundtruth

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    delta: float = 1,
    gamma: float = 2,
    A: float = 1,
    reduction: str = "none",
):

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    loss = ce_loss * torch.where(targets == 1,
                                 (1-p)**gamma, # foreground
                                A*(1-targets)**delta * p**gamma)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class CenterDirectionLoss(nn.Module):

    def __init__(self, center_model, enable_direction_loss=True, regression_loss='l1', enable_localization_loss=False,
                 localization_loss='l1', MAX_OFFSET=1024, **kargs):
        super().__init__()

        self.enable_localization_loss = enable_localization_loss    # enable learning of localization network
        self.enable_direction_loss = enable_direction_loss          # enable learning of center direction regression network

        self.MAX_OFFSET = MAX_OFFSET                                # max image size

        ################################################################
        # Prepare all loss functions
        def abs_jit(X,Y): return torch.abs(X - Y)
        def mse_jit(X,Y): return torch.pow(X - Y, 2)
        def m4e_jit(X, Y): return torch.pow(X - Y, 4)

        self.loss_abs_fn = abs_jit
        self.loss_mse_fn = mse_jit
        self.loss_m4e_fn = m4e_jit

        self.loss_hinge_fn = lambda X, Y, sign_fn, eps=0: (torch.clamp_min(sign_fn(Y - X), eps) - eps)
        self.loss_smoothL1_fn = lambda X,Y,beta,pow: torch.where((X - Y).abs() < beta,
                                                                 torch.pow(X - Y, pow) / (pow * beta),
                                                                 (X - Y).abs() - 1/float(pow) * beta)
        self.loss_inverted_smoothL1_fn = lambda X,Y,beta,pow: torch.where((X - Y).abs() > beta,
                                                                          torch.pow(X - Y, pow) / (pow * beta),
                                                                          (X - Y).abs() - 1/float(pow) * beta)
        self.loss_bce_logits = torch.nn.BCEWithLogitsLoss(reduction='none')

        def construct_loss(loss_type):
            args = {}
            if type(loss_type) is dict:
                args = loss_type['args'] if 'args' in loss_type else {}
                loss_type = loss_type['type']

            if loss_type.upper() in ['L1','MAE']:
                return partial(self.loss_abs_fn,**args)
            elif loss_type.upper() in ['L2', 'MSE']:
                return partial(self.loss_mse_fn,**args)
            elif loss_type.lower() in ['hinge']:
                return partial(self.loss_hinge_fn,**args)
            elif loss_type.lower() in ['smoothl1']:
                return partial(self.loss_smoothL1_fn,**args)
            elif loss_type.lower() in ['inverted-smoothl1']:
                return partial(self.loss_inverted_smoothL1_fn,**args)
            elif loss_type.lower() in ['cross-entropy','bce']:
                return partial(self.loss_bce_logits,**args)
            elif loss_type.lower() in ['focal']:
                return lambda X,Y: sigmoid_focal_loss(X,Y,reduction="none",**args)
            else:
                raise Exception('Unsuported loss type: \'%s\'' % loss_type)

        self.regression_loss_fn = construct_loss(regression_loss)
        self.localization_loss_fn = construct_loss(localization_loss)

        self.center_model = center_model


    def forward(self, prediction, instances, labels, centerdir_responses=None, centerdir_gt=None, ignore_mask=None,
                w_cos=1, w_sin=1, w_fg=1, w_bg=1, w_cent=1, w_fg_cent=1, w_bg_cent=1, reduction_dims=(1,2,3)):

        batch_size, height, width = prediction.size(0), prediction.size(2), prediction.size(3)

        loss_output_shape = [d for i,d in enumerate(prediction.shape) if i not in reduction_dims]
        loss_zero_init = lambda: torch.zeros(size=loss_output_shape,device=prediction.device)

        loss_sin, loss_cos = map(torch.clone,[loss_zero_init()]*2)

        loss_centers = loss_zero_init()

        # batch computation ---
        labels = labels.unsqueeze(1)
        bg_mask = labels == 0
        fg_mask = bg_mask == False

        prediction_sin = prediction[:, 0].unsqueeze(1)
        prediction_cos = prediction[:, 1].unsqueeze(1)

        if instances.dtype != torch.int16:
            instances = instances.type(torch.int16)

        # mark ignore regions as -9999 in instances so that size can be correctly calculated
        if ignore_mask is not None:
            instances = instances.clone() # do not destroy original
            instances[ignore_mask.squeeze(dim=1) == 1] = -9999

        # count number of pixels per instance
        instance_ids, instance_sizes = instances.reshape(instances.shape[0], -1).unique(return_counts=True, dim=-1)

        # count number of instance for each batch element (without background and ignored regions)
        num_instances = sum([len(set(ids.unique().cpu().numpy()) - set([0,-9999])) for ids in instance_ids])
        num_bg_pixels = instance_sizes.repeat(batch_size, 1)[instance_ids == 0].sum().float()

        # retrieve groundtruth values (either computed or from cache)
        gt_R, gt_theta, gt_sin_th, gt_cos_th, gt_center_mask = CenterDirGroundtruth.parse_groundtruth(centerdir_gt)

        with torch.no_grad():

            def _init_weights_for_instances(W, group_instance, group_instance_ids, group_instance_sizes,
                                            _num_bg_pixels=num_bg_pixels, _num_hard_negative_pixels=torch.tensor(0.0)):
                # ensure each instance (and background) is weighted equally regardless of pixels size
                with torch.no_grad():
                    num_instances = sum([len(set(ids.unique().cpu().numpy()) - set([0, -9999])) for ids in group_instance])
                    for b in range(batch_size):
                        for id in group_instance_ids[b].unique():
                            mask_id = group_instance[b].eq(id).unsqueeze(0)
                            if id == 0:
                                # for BG instance we normalize based on the number of all bg pixels over the whole batch
                                instance_normalization = _num_bg_pixels*1
                                instance_normalization = instance_normalization*(3/1.0 if _num_hard_negative_pixels > 0 else 2)
                            elif id < 0:
                                if _num_hard_negative_pixels > 0:
                                    # for hard-negative instances we normalized based on number of them (in pixels)
                                    instance_normalization = _num_hard_negative_pixels*torch.log(_num_hard_negative_pixels+1)
                                    instance_normalization = instance_normalization * 3 / 1.0
                                else:
                                    instance_normalization = 1.0
                            else:
                                # for FG instances we normalized based on the size of instance (in pixel) and the number of
                                # instances over the whole batch
                                instance_pixels = group_instance_sizes[group_instance_ids[b] == id].sum().float()
                                instance_normalization = instance_pixels * num_instances *1
                                instance_normalization = instance_normalization*(3/1.0 if _num_hard_negative_pixels > 0 else 2)

                            # BG and FG are treated as equal so add multiplication by 2 (or 3 if we also have hard-negatives)
                            # instance_normalization = instance_normalization * _N
                            W[b][mask_id] *= 1.0 / instance_normalization
                return W

        ######################################################
        ### loss for estimating center from center directions outputs

        if self.enable_localization_loss:
            with torch.no_grad():
                assert centerdir_responses is not None

                _, center_heatmap = centerdir_responses

                centers_hard_neg_mask = torch.ones_like(gt_center_mask, requires_grad=False, device=prediction.device, dtype=torch.float32)
                centers_hard_neg_mask *= 1.0 / (height * width * batch_size)

                if ignore_mask is not None:
                    centers_hard_neg_mask *= 1 - ignore_mask.type(centers_hard_neg_mask.type())

                if w_fg_cent != 1:
                    centers_hard_neg_mask[gt_center_mask > 0] *= w_fg_cent

                if w_bg_cent != 1:
                    centers_hard_neg_mask[gt_center_mask <= 0] *= w_bg_cent

            loss_centers += torch.sum(centers_hard_neg_mask * self.localization_loss_fn(center_heatmap.unsqueeze(1),
                                                                                        gt_center_mask), dim=reduction_dims)

        ######################################################
        ### direction vector losses (cos, sin)
        if self.enable_direction_loss:

            with torch.no_grad():

                mask_weights = torch.ones_like(prediction_sin, requires_grad=False, device=prediction.device)

                mask_weights[fg_mask] = w_fg
                mask_weights[bg_mask] = w_bg

                if ignore_mask is not None:
                    mask_weights *= 1 - ignore_mask.type(mask_weights.type())

                mask_weights = _init_weights_for_instances(mask_weights, instances, instance_ids, instance_sizes)

            # add regression loss for sin(x), cos(x)
            loss_sin += torch.sum(mask_weights * self.regression_loss_fn(prediction_sin, gt_sin_th), dim=reduction_dims)
            loss_cos += torch.sum(mask_weights * self.regression_loss_fn(prediction_cos, gt_cos_th), dim=reduction_dims)

        loss_sin = w_sin * loss_sin
        loss_cos = w_cos * loss_cos

        loss_centers = w_cent * loss_centers

        loss_direction_total = loss_sin + loss_cos

        # total/final loss:
        loss = loss_direction_total + loss_centers

        # add epsilon as a way to force values to tensor/cuda
        eps = prediction.sum() * 0
        # convert all losses to tensors to ensure proper parallelization with torch.nn.DataParallel
        losses = [t + eps for t in [loss, loss_direction_total, loss_centers,
                                    loss_sin, loss_cos]]

        return tuple(losses)
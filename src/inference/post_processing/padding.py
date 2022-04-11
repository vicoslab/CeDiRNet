import numpy as np

import torch

import scipy

from utils.transforms import Padding

class PaddingPostProcesser:
    '''
    A wrapper over main inference processing class that is able to handle images with added padding border.

    Class is able to handle images that have edge-replication or mirror/symmetric padding added by the dataset transform
    function. Detections in padding area are removed based on padding type:
     - edge-replication padding: clip location of object to within valid image region
     - mirror/symmetric padding: re-project detection back to original image space and merge them with existing detection

    Additionally, any detection more than 'max_border_distance' away from the original image border is removed.
    '''

    def __init__(self, main_processer, max_border_distance=0.01):
        self.main_processer = main_processer

        self.max_border_distance = max_border_distance

        # add this in case any other wrapper needs it
        self.device = main_processer.device

    def get_center_model_list(self):
        return self.main_processer.get_center_model_list()

    def clean_memory(self):
        self.main_processer.clean_memory()

    def __call__(self, *args, **kwargs):

        for input, output in self.main_processer(*args, **kwargs):
            predictions = output['predictions']
            pred_mask = output['pred_mask']
            ignore = input['ignore']

            if ignore is not None and len(predictions) > 0:
                # If images had extra padding added then (TODO: remove padding from final evaluation images and) if border replication
                # was used (edge or symmetric/mirror) then translate any detections from padding region back to original
                # image space (and merge it with any original detections)

                for_removal_idx = []
                invalid_outer_detections = np.zeros(len(predictions), dtype=np.bool)

                out_columns = predictions.shape[1]

                ############################################################################################################
                # Handle edge-replication padding
                padding_edge_map = ignore & Padding.IGNORE_PADDING_EDGE > 0
                if padding_edge_map.any():
                    region_idx = torch.nonzero(~padding_edge_map)
                    _, _, t, l = region_idx.min(axis=0)[0].cpu().numpy()
                    _, _, b, r = region_idx.max(axis=0)[0].cpu().numpy()

                    # remove detection that are too far away from actual border
                    if self.max_border_distance and self.max_border_distance > 0:
                        B = self.max_border_distance
                        invalid_outer_detections |= (predictions[:, 1] < t - B) | (predictions[:, 1] > b + B) | \
                                                    (predictions[:, 0] < l - B) | (predictions[:, 0] > r + B)

                    # with simple edge replication then only clamp detections into valid image region
                    predictions[:, 1] = np.clip(predictions[:, 1], t, b)
                    predictions[:, 0] = np.clip(predictions[:, 0], l, r)

                ############################################################################################################
                # Handle mirror/symmetric padding
                padding_symmetric_map = ignore & Padding.IGNORE_PADDING_SYMMETRIC > 0
                if padding_symmetric_map.any():
                    region_idx = torch.nonzero(~padding_symmetric_map)
                    _, _, t, l = region_idx.min(axis=0)[0].cpu().numpy()
                    _, _, b, r = region_idx.max(axis=0)[0].cpu().numpy()

                    # remove detection that are too far away from actual border
                    if self.max_border_distance and self.max_border_distance > 0:
                        B = self.max_border_distance
                        invalid_outer_detections |= (predictions[:, 1] < t - B) | (predictions[:, 1] > b + B) | \
                                                    (predictions[:, 0] < l - B) | (predictions[:, 0] > r + B)

                    # with mirror/symmetric replication project detections back to original image space
                    top_idx, bottom_idx = predictions[:, 1] < t, predictions[:, 1] > b
                    left_idx, right_idx = predictions[:, 0] < l, predictions[:, 0] > r

                    predictions[top_idx, 1] = 2 * t - predictions[top_idx, 1]
                    predictions[bottom_idx, 1] = 2 * b - predictions[bottom_idx, 1]
                    predictions[left_idx, 0] = 2 * l - predictions[left_idx, 0]
                    predictions[right_idx, 0] = 2 * r - predictions[right_idx, 0]

                    outer_detections = top_idx | bottom_idx | left_idx | right_idx

                    if outer_detections.any() and not outer_detections.all():
                        # remove any "outer" detection if other true detection is less than 20px away
                        dist = scipy.spatial.distance_matrix(predictions[outer_detections, :2],
                                                             predictions[~outer_detections, :2]).min(axis=1)

                        for_removal_idx += list(np.where(outer_detections)[0][dist < 20])

                if len(for_removal_idx) > 0 or any(invalid_outer_detections):
                    selected_pred_idx = [i for i in range(len(predictions)) if
                                         i not in for_removal_idx and not invalid_outer_detections[i]]

                    # filter-out ones that are matched
                    predictions = predictions[selected_pred_idx, :]

                    new_pred_mask = torch.zeros_like(pred_mask)
                    for new_id, old_id in enumerate(selected_pred_idx):
                        new_pred_mask[pred_mask == old_id + 1] = new_id + 1
                    pred_mask = new_pred_mask

                    # padding_map = padding_symmetric_map and padding_edge_map
                    # if padding_map.any():
                    #     # crop image and all outputs to valid image region
                    #     region_idx = np.where(~padding_map)
                    #     t, b, l, r = region_idx[0].min(), region_idx[0].max(), region_idx[1].min(), region_idx[1].max()

                predictions = predictions.reshape(-1, out_columns)

                output.update(dict(predictions=predictions,
                                   pred_mask=pred_mask))

            yield input, output
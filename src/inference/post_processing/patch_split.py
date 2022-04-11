import numpy as np

import torch

from utils.utils import ImageGridCombiner

class PatchSplitPostProcesser:
    '''
    A wrapper over main inference processing class for center directions models that is able to handle images split into smaller
    patches.

    Class is able to handle images that have been split into patches by PatchSplitDataset, which are identified using
    'grid_index' key in input/sample array. Results from individual patches are then merged into a single  image
    using utils.ImageGridCombiner().

    Class is able to handle the following input/output keys:
     - input keys: 'image', 'im_name', 'centerdir_groundtruth', 'ignore', 'instance', 'instance_ids', 'center_dict'
     - output keys: 'output', 'predictions', 'pred_mask', 'pred_mask_ids', 'pred_heatmap'

    '''

    HEATMAP_INPUT_KEYS = ['image', 'centerdir_groundtruth', 'ignore']
    HEATMAP_OUTPUT_KEYS = ['output', ]

    def __init__(self, main_processer, enabled=True, predictions_merge_thr=0.01):
        self.main_processer = main_processer

        self.enabled = enabled
        self.predictions_merge_thr = predictions_merge_thr

        # add this in case any other wrapper needs it
        self.device = main_processer.device

    def get_center_model_list(self):
        return self.main_processer.get_center_model_list()

    def clean_memory(self):
        self.main_processer.clean_memory()

    def __call__(self, *args, **kwargs):

        grid_combiner = {}
        grid_index = None

        # prepare separate buffers for each center model that is being processed
        for center_model_desc in self.get_center_model_list():
            center_model_name = center_model_desc['name']
            grid_combiner[center_model_name] = ImageGridCombiner()


        for input,output in self.main_processer(*args, **kwargs):

            if 'grid_index' not in input or not self.enabled:
                ######################################################################
                # IN NORMAL PROCESSING - just yield and continue
                yield input, output
            else:
                ######################################################################
                # IN PATCH-SPLIT PROCESSING
                center_model_name = output['center_model_name']
                grid_index = input['grid_index']

                im_name = input['im_name']
                center_dict = input['center_dict']
                instance = input['instance']
                predictions = output['predictions']
                pred_mask = output['pred_mask']
                pred_heatmap = output['pred_heatmap']

                # collect results (tensor data) from all images that have the same grid_index N
                partial_data_tensor = dict()
                partial_data_tensor.update({k: input[k] for k in self.HEATMAP_INPUT_KEYS})
                partial_data_tensor.update({k: output[k] for k in self.HEATMAP_OUTPUT_KEYS})
                partial_data_tensor.update({('pred_heatmap_%d' % i): c for i, c in enumerate(pred_heatmap)})

                # use logical OR when merging ignore_flags (to preserve original flags)
                custom_merge_ops = dict(ignore_flags=lambda Y, X: X | Y)

                partial_data_instance_desc = dict(centers=(center_dict, instance, True),
                                                  predictions=({i + 1: pred for i, pred in enumerate(predictions)},
                                                               pred_mask, False))

                # send data to grid_combiner for merging
                # when new index is detected it will merge and return previously collected data
                finished_image = grid_combiner[center_model_name].add_image(im_name,
                                                                            grid_index[0].cpu().numpy(),
                                                                            partial_data_tensor,
                                                                            partial_data_instance_desc,
                                                                            custom_merge_ops=custom_merge_ops)

                # if this is final image then extract merged result and return it
                if finished_image is not None:
                    yield self._extract_merged_results(center_model_name, finished_image,
                                                       instance_dtype=instance.dtype,
                                                       pred_mask_dtype=pred_mask.dtype,
                                                       num_heatmap=len(pred_heatmap))


        # need to additionally handle last sample when in patch-split
        if grid_index is not None:
            for center_model_name, grid_combiner_ in grid_combiner.items():
                yield self._extract_merged_results(center_model_name, grid_combiner_.current_data,
                                                   instance_dtype=instance.dtype,
                                                   pred_mask_dtype=pred_mask.dtype,
                                                   num_heatmap=len(pred_heatmap))

    def _extract_merged_results(self, center_model_name, data, instance_dtype, pred_mask_dtype, num_heatmap):
        im_name = data.get_image_name()

        input = dict(zip(self.HEATMAP_INPUT_KEYS, data.get(self.HEATMAP_INPUT_KEYS)))
        output = dict(zip(self.HEATMAP_OUTPUT_KEYS, data.get(self.HEATMAP_OUTPUT_KEYS)))

        im_shape = input['image'].shape[-2:]

        pred_heatmap = data.get(['pred_heatmap_%d' % i for i in range(num_heatmap)])

        instances_ = torch.zeros(size=im_shape, dtype=instance_dtype, device=self.main_processer.device)
        center_dict, instance, instance_ids = data.get_instance_description('centers', out_mask_tensor=instances_, overlap_thr=0.0)

        pred_mask_ = torch.zeros(size=im_shape, dtype=pred_mask_dtype, device=self.main_processer.device)
        predictions, pred_mask, pred_mask_ids = data.get_instance_description('predictions', out_mask_tensor=pred_mask_,
                                                                              overlap_thr=self.predictions_merge_thr)

        predictions = np.array([predictions[k] for k in sorted(predictions.keys())])

        input.update(dict(im_name=im_name,
                          instance=instance,
                          instance_ids=instance_ids,
                          center_dict=center_dict))

        output.update(dict(predictions=predictions,
                           pred_mask=pred_mask,
                           pred_mask_ids=pred_mask_ids,
                           pred_heatmap=pred_heatmap,
                           center_model_name=center_model_name))

        return input, output

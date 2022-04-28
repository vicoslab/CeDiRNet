import os

import numpy as np
from tqdm import tqdm

from models.center_groundtruth import CenterDirGroundtruth

import torch

class CeDiRNetProcesser:
    '''
    Main inference processing class for CeDiRNet models (with center detections).

    When main function is call it will iterate over each image from the provided dataset iterator and process them with
    provided self.model and self.center_model.

    Class is able to handle the following items in the input data (sample from the dataset):
      - required keys: image, im_name, instance,
      - optional keys: centerdir_groundtruth, ignore, center, grid_index,

    The following keys are emitted for each processed/merged image:
      - as reference input data:  <ANY ORIGINAL SAMPLE DATA such as image, im_name, instance, center, centerdir_groundtruth, ignore etc. > + center_dict
      - as processed output data: output, output_cls, resnet_features, CAM, predictions, pred_mask, pred_heatmap
    '''
    def __init__(self, model, center_model_list, device=None):
        self.model = model

        self.center_model_list = center_model_list

        self.device = device

    def get_center_model_list(self):
        return self.center_model_list

    def clean_memory(self):
        self.model.cpu()
        self.model = None

        if self.center_model_list is not None:
            for center_model_desc in self.get_center_model_list():
                center_model_desc['model'].cpu()
                center_model_desc['model'] = None

    def __call__(self, dataset_it, centerdir_groundtruth_op=None, tqdm_kwargs={}):

        assert self.model is not None
        self.model.eval()

        for center_model_desc in self.get_center_model_list():
            assert center_model_desc['model'] is not None

            center_model_desc['model'].eval()

        im_image = 0

        for sample_ in tqdm(dataset_it, **tqdm_kwargs):

            # call centerdir_groundtruth_op first which will create any missing centerdir_groundtruth (using GPU) and add synthetic output
            # and also update model function if requested to use cached outputs (or requested to save them)
            if centerdir_groundtruth_op is not None:
                sample_ = centerdir_groundtruth_op(sample_, torch.arange(0, dataset_it.batch_size).int())
                model = centerdir_groundtruth_op.module.insert_cached_model(self.model, sample_)
            else:
                model = self.model

            im_batch = sample_['image']

            output_batch_ = model(im_batch)

            for center_model_desc in self.get_center_model_list():
                center_model_name = center_model_desc['name']
                center_model = center_model_desc['model']

                center_output = center_model(output_batch_, detect_centers=True, **sample_)

                output_batch, center_pred, center_heatmap = [center_output[k] for k in ['output', 'center_pred', 'center_heatmap']]

                if 'centerdir_groundtruth' in sample_:
                    # get gt_centers from polar_gt and convert them to dictionary (filter-out non visible and ignored examples)
                    # if ignore_flags is present then set to remove all groundtruths where ONLY ignore flag (encoded as 1) is present but not others
                    # (do not remove other types such as truncated, overlap border, difficult)
                    instances = sample_['instance'].squeeze(dim=1)
                    center_ignore = sample_['ignore'] == 1 if 'ignore' in sample_ else None

                    _, _, _, _, gt_centers, _, _, _ = CenterDirGroundtruth.parse_polar_groundtruth(sample_['centerdir_groundtruth'])
                    gt_centers_dict = CenterDirGroundtruth.convert_gt_centers_to_dictionary(gt_centers,
                                                                                            instances=instances,
                                                                                            ignore=center_ignore)
                else:
                    gt_centers_dict = None

                sample_keys = sample_.keys()

                for batch_i in range(min(dataset_it.batch_size, len(sample_['im_name']))):
                    im_image += 1
                    output = output_batch[batch_i:batch_i + 1]

                    sample = {k: sample_[k][batch_i:batch_i + 1] for k in sample_keys}

                    im_name = sample['im_name'][0]
                    base, _ = os.path.splitext(os.path.basename(im_name))

                    instance = sample['instance'].squeeze()
                    ignore = sample.get('ignore')

                    if 'centerdir_groundtruth' in sample_:
                        sample['centerdir_groundtruth'] = sample_['centerdir_groundtruth'][0][batch_i]

                    if gt_centers_dict is not None:
                        center_dict = gt_centers_dict[batch_i]

                        # manually remove instance that have been ignored
                        if ignore is not None:
                            for id in instance.unique():
                                id = id.item()
                                if id > 0 and id not in center_dict.keys():
                                    instance[instance == id] = 0
                    else:
                        center_dict = None

                    # extract prediction heatmap and sorted prediction list
                    pred_heatmap = torch.relu(center_heatmap[batch_i].unsqueeze(0).unsqueeze(0))
                    predictions = center_pred[batch_i][center_pred[batch_i,:,0]==1][:,1:].cpu().numpy()

                    idx = np.argsort(predictions[:, -1])
                    predictions = predictions[idx[::-1], :]

                    # simply return all input and output data
                    output_dict = dict(output=output,
                                       predictions=predictions,
                                       pred_mask=torch.zeros_like(pred_heatmap),
                                       pred_heatmap=pred_heatmap,
                                       center_model_name=center_model_name)

                    # update sample data with center_dict and override im_name and instance
                    sample.update(dict(im_name=im_name, # override to remove dimension
                                       instance=instance, # override to remove dimension
                                       center_dict=center_dict,))

                    yield sample, output_dict


import os

import torch
import numpy as np
import json

from utils.evaluation import NumpyEncoder
from utils.evaluation.center_global_min import CenterGlobalMinimizationEval

class PatchMergeCenterGlobalMinimizationEval(CenterGlobalMinimizationEval):
    def __init__(self, merge_threshold_px=40, save_raw_results=True, only_save_raw_results=False, *args, **kwargs):
        super(PatchMergeCenterGlobalMinimizationEval,self).__init__(*args, **kwargs)

        self.center_ap_eval = None # disable
        self.merge_threshold_px = merge_threshold_px
        self.only_save_raw_results = only_save_raw_results
        self.save_raw_results = save_raw_results

        self.total_predictions_and_score = np.zeros((0,3),dtype=np.float32)
        self.total_gt_centers = np.zeros((0,2),dtype=np.float32)
        self.image_count = 0

    def save_str(self):
        return "tau=%.1f-d_alpha=%.1f-score_thr=%.1f-center_ap_eval=%s" % (self.tau_thr, self.merge_threshold_px, self.score_thr, str(self.center_ap_eval))

    @classmethod
    def get_image_global_location(cls, im_name):
        loc_dict = {}
        for loc_keyval in os.path.splitext(im_name)[0].split("_"):
            if "=" not in loc_keyval:
                continue
            key_val = loc_keyval.split("=")
            if len(key_val) != 2:
                continue
            loc_dict[key_val[0]] = float(key_val[1])
        return loc_dict

    def add_image_prediction(self, im_name, im_index, im_shape, predictions, predictions_score,
                             gt_instances, gt_centers_dict, gt_difficult, return_matched_gt_idx=False):
        # only translate predictions and groundtruths to original image space
        global_loc = self.get_image_global_location(im_name)

        assert 'i' in global_loc and 'j' in global_loc, "Cannot calculate global location: missing 'i' and/or 'j' in image name!"

        i,j = global_loc['i'],global_loc['j']

        if len(predictions) > 0:
            # convert locations into global image space
            predictions_and_score = np.concatenate((predictions[:, :2] + np.array([i,j]).reshape(1,2),
                                                    predictions_score.reshape(-1, 1)), axis=1)

            self.total_predictions_and_score = np.concatenate((self.total_predictions_and_score, predictions_and_score), axis=0)

        if len(gt_centers_dict) > 0:
            gt_centers = [gt_centers_dict[k] for k in sorted(gt_centers_dict.keys())]
            gt_centers = np.array(gt_centers) + np.array([j, i])

            self.total_gt_centers = np.concatenate((self.total_gt_centers, gt_centers),axis=0)

        self.image_count += 1
        return 0,0,[]

    def calc_and_display_final_metrics(self, print_result=True, plot_result=True, save_dir=None,
                                       load_raw_results=False, **kwargs):

        def _remove_duplicates_idx(sample_loc, thr, sample_loc_score=None):
            if len(sample_loc) <= 0:
                return None
            grouped_samples = self.split_samples_into_groups(sample_loc, np.array([]), distance_thr=thr, grid_count=(16, 16))

            selected_samples = np.ones((len(sample_loc),), dtype=np.bool)

            for group_sample_idx, _ in grouped_samples:
                # sort predictions by desending score so that best predictions will not be suppressed
                if sample_loc_score is not None:
                    sort_idx = np.argsort(sample_loc_score[group_sample_idx])
                    group_sample_idx = group_sample_idx[sort_idx[::-1]]

                group_samples = torch.from_numpy(sample_loc[group_sample_idx, :2])

                # suppress predictions that are closer than self.merge_threshold_px
                dist_matric = torch.cdist(group_samples, group_samples).cpu().numpy() <= thr

                for i in range(len(group_sample_idx)):
                    i_group_sample_idx = group_sample_idx[i + 1:len(group_sample_idx)]
                    invalid_idx = np.where(dist_matric[i, i + 1:len(group_sample_idx)] & \
                                           selected_samples[i_group_sample_idx])
                    selected_samples[i_group_sample_idx[invalid_idx]] = False

            return selected_samples

        # remove duplicated predictions
        retain_idx = _remove_duplicates_idx(self.total_predictions_and_score[:, :2], self.merge_threshold_px,
                                            self.total_predictions_and_score[:, -1])
        if retain_idx is not None:
            self.total_predictions_and_score = self.total_predictions_and_score[retain_idx,:]

        # remove duplicated groundtruths
        MERGER_GT_DIST = 5
        retain_idx = _remove_duplicates_idx(self.total_gt_centers, MERGER_GT_DIST)
        if retain_idx is not None:
            self.total_gt_centers = self.total_gt_centers[retain_idx,:]

        gt_instances_dict = {i:c for i,c in enumerate(self.total_gt_centers)}

        raw_detections = dict(predictions=self.total_predictions_and_score[:, :2],
                              predictions_score=self.total_predictions_and_score[:, -1],
                              gt_centers_dict=gt_instances_dict,
                              image_count=self.image_count)


        # save results for debugging
        if save_dir is not None:
            out_dir = os.path.join(save_dir, self.exp_name, self.save_str())
            os.makedirs(out_dir, exist_ok=True)

            import pickle
            if load_raw_results:
                with open(os.path.join(out_dir, 'raw_detections.pkl'),'rb') as f:
                    raw_detections = pickle.load(f)
            else:
                if self.save_raw_results:
                    with open(os.path.join(out_dir, 'raw_detections.pkl'),'wb') as f:
                        pickle.dump(raw_detections, f)
                if self.only_save_raw_results:
                    return dict()

        res = super(PatchMergeCenterGlobalMinimizationEval,self).add_image_prediction(None,None,None,
                    predictions=raw_detections['predictions'], predictions_score=raw_detections['predictions_score'],
                    gt_instances=None, gt_centers_dict=raw_detections['gt_centers_dict'], gt_difficult=None,return_matched_gt_idx=True)


        Re = np.array(self.metrics['Re']).mean()
        mae = np.array(self.metrics['mae']).sum()/raw_detections['image_count']
        rmse = np.array(self.metrics['rmse']).sum()/raw_detections['image_count']
        ratio = np.array(self.metrics['ratio']).mean()
        AP = np.array(self.metrics['precision']).mean()
        AR = np.array(self.metrics['recall']).mean()
        F1 = np.array(self.metrics['F1']).mean()

        if print_result:
            print('Re=%.4f, mae=%.4f, rmse=%.4f, ratio=%.4f, AP=%.4f, AR=%.4f, F1=%.4f' % (Re, mae, rmse, ratio, AP, AR, F1))

        metrics = dict(AP=AP, AR=AR, F1=F1, ratio=ratio, Re=Re, mae=mae, rmse=rmse, all_images=self.metrics, metrics_mAP=None)

        ########################################################################################################
        # SAVE EVAL RESULTS TO JSON FILE
        if save_dir is not None:
            out_dir = os.path.join(save_dir, self.exp_name, self.save_str())
            os.makedirs(out_dir, exist_ok=True)

            if metrics is not None:
                with open(os.path.join(out_dir, 'results.json'), 'w') as file:
                    file.write(json.dumps(metrics, cls=NumpyEncoder))

        return None, metrics

    def get_results_timestamp(self, save_dir):
        if self.only_save_raw_results:
            res_filename = os.path.join(save_dir, self.exp_name, self.save_str(), 'raw_detections.pkl')
        else:
            res_filename = os.path.join(save_dir, self.exp_name, self.save_str(), 'results.json')


        return os.path.getmtime(res_filename) if os.path.exists(res_filename) else 0

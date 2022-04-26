#!/usr/bin/python

import os

from matplotlib import pyplot as plt

import numpy as np

import torch

from datasets import get_dataset
from models import get_model, get_center_model
from utils.utils import Visualizer
from utils.evaluation.center import CenterEvaluation
from utils.evaluation.center_global_min import CenterGlobalMinimizationEval
from utils.evaluation.center_global_min_patch import PatchMergeCenterGlobalMinimizationEval

from inference.processing import CeDiRNetProcesser
from inference.post_processing.padding import PaddingPostProcesser
from inference.post_processing.patch_split import PatchSplitPostProcesser
from inference.post_processing.corss_validation import CrossValidationPostProcesser

class Evaluator:
    def __init__(self, args):
        if True:
            plt.ion()
        else:
            plt.ioff()
            plt.switch_backend("agg")

        if args.get('cudnn_benchmark'):
            torch.backends.cudnn.benchmark = True

        self.args = args

        # set device
        self.device = torch.device("cuda:0" if args['cuda'] else "cpu")

    def initialize(self):
        args = self.args

        ###################################################################################################
        # set dataset

        if args.get('cross_validation_folds'):
            # cross-validation: construct separate dataset and models for each fold
            fold_arrays = [self._construct_dataset_and_processing(args, self.device, fold) for fold in args.get('cross_validation_folds')]
            self.dataset_it, self.centerdir_groundtruth_op, self.processed_image_iter = zip(*fold_arrays)

            self.processed_image_iter = CrossValidationPostProcesser(self.processed_image_iter)
        else:
            # default processing
            self.dataset_it, self.centerdir_groundtruth_op, \
                self.processed_image_iter = self._construct_dataset_and_processing(args, self.device)

        ###################################################################################################
        # Visualizer
        self.visualizer = Visualizer(('image', 'centers'), to_file_only=args.get('display_to_file_only'))

    @classmethod
    def _construct_dataset_and_processing(self, args, device, fold=None):

        if fold is not None:
            args['dataset']['kwargs']['fold'] = fold

            args['checkpoint_path'] = args['checkpoint_path'].replace("fold=XYZ","fold=%d" % fold)
            center_checkpoint_path = args.get('center_checkpoint_path')
            if center_checkpoint_path is not None:
                if isinstance(center_checkpoint_path,list):
                    args['center_checkpoint_path'] = [p.replace("fold=XYZ", "fold=%d" % fold) for p in center_checkpoint_path]
                else:
                    args['center_checkpoint_path'] = center_checkpoint_path.replace("fold=XYZ","fold=%d" % fold)

        ###################################################################################################
        # dataloader
        dataset_workers = args['dataset']['workers'] if 'workers' in args['dataset'] else 0
        dataset_batch = args['dataset']['batch_size'] if 'batch_size' in args['dataset'] else 1

        dataset, centerdir_groundtruth_op = get_dataset(args['dataset']['name'], args['dataset']['kwargs'],
                                                        args['dataset'].get('centerdir_gt_opts'))

        if centerdir_groundtruth_op is not None:
            centerdir_groundtruth_op = torch.nn.DataParallel(centerdir_groundtruth_op).to(device)

        dataset_it = torch.utils.data.DataLoader(
            dataset, batch_size=dataset_batch, shuffle=False, drop_last=False, num_workers=dataset_workers, pin_memory=True if args['cuda'] else False)

        ###################################################################################################
        # load model
        model = get_model(args['model']['name'], args['model']['kwargs'])
        model = torch.nn.DataParallel(model).to(device)

        center_model_list = []

        def get_center_fn():
            return get_center_model(args['center_model']['name'], args['center_model']['kwargs'])

        # prepare center_model based on number of center_checkpoint_path that will need to be processed
        if args.get('center_checkpoint_path') and isinstance(args['center_checkpoint_path'],list):
            assert 'center_checkpoint_name_list' in args and isinstance(args['center_checkpoint_name_list'],list)
            assert len(args['center_checkpoint_name_list']) == len(args['center_checkpoint_path'])

            for center_checkpoint_name, center_checkpoint_path in zip(args['center_checkpoint_name_list'], args['center_checkpoint_path']):
                center_model = get_center_fn()
                center_model_list.append(dict(name=center_checkpoint_name,
                                              checkpoint=center_checkpoint_path,
                                              model=center_model))
        else:
            center_checkpoint_name = args.get('center_checkpoint_name') if 'center_checkpoint_name' in args else ''
            center_checkpoint_path = args.get('center_checkpoint_path')

            center_model = get_center_fn()
            center_model_list.append(dict(name=center_checkpoint_name,
                                          checkpoint=center_checkpoint_path,
                                          model=center_model))

        for center_model_desc in center_model_list:
            center_model_desc['model'] = torch.nn.DataParallel(center_model_desc['model']).to(device)

        ###################################################################################################
        # load snapshot
        if os.path.exists(args['checkpoint_path']):
            print('Loading from "%s"' % args['checkpoint_path'])
            state = torch.load(args['checkpoint_path'])
            if 'model_state_dict' in state:
                if 'module.model.segmentation_head.2.weight' in state['model_state_dict']:
                    checkpoint_input_weights = state['model_state_dict']['module.model.segmentation_head.2.weight']
                    checkpoint_input_bias = state['model_state_dict']['module.model.segmentation_head.2.bias']
                    model_output_weights = model.module.model.segmentation_head[2].weight
                    if checkpoint_input_weights.shape != model_output_weights.shape:
                        state['model_state_dict']['module.model.segmentation_head.2.weight'] = checkpoint_input_weights[:2, :, :, :]
                        state['model_state_dict']['module.model.segmentation_head.2.bias'] = checkpoint_input_bias[:2]
                        print('WARNING: #####################################################################################################')
                        print('WARNING: regression output shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                        print('WARNING: #####################################################################################################')

                model.load_state_dict(state['model_state_dict'], strict=True)
            if not args.get('center_checkpoint_path') and 'center_model_state_dict' in state:
                for center_model_desc in center_model_list:
                    center_model_desc['model'].load_state_dict(state['center_model_state_dict'], strict=False)
        else:
            raise Exception('checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

        for center_model_desc in center_model_list:
            if center_model_desc['checkpoint'] is None:
                continue
            if os.path.exists(center_model_desc['checkpoint']):
                print('Loading center model from "%s"' % center_model_desc['checkpoint'])
                state = torch.load(center_model_desc['checkpoint'])
                if 'center_model_state_dict' in state:
                    if 'module.instance_center_estimator.conv_start.0.weight' in state['center_model_state_dict']:
                        checkpoint_input_weights = state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight']
                        center_input_weights = center_model_desc['model'].module.instance_center_estimator.conv_start[0].weight
                        if checkpoint_input_weights.shape != center_input_weights.shape:
                            state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight'] = checkpoint_input_weights[:,:2,:,:]

                            print('WARNING: #####################################################################################################')
                            print('WARNING: center input shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                            print('WARNING: #####################################################################################################')

                    center_model_desc['model'].load_state_dict(state['center_model_state_dict'], strict=False)
            else:
                raise Exception('checkpoint_path {} does not exist!'.format(center_model_desc['checkpoint']))

        ###################################################################################################
        # MAIN PROCESSING PIPELINE:
        #
        #   dataset ->
        #       -> CeDiRNetProcesser (model + center detection) ->
        #       -> (optional) PaddingPostProcesser (handles detections in padding region) ->
        #       -> PatchSplitPostProcesser (merges results from patch-split dataset back into original space)
        #       -> [THIS evaluator]

        # main inference processer for center directions models
        processed_image_iter = CeDiRNetProcesser(model, center_model_list, device)

        # add post-processer for handling padding if needed (will remove/handle detections in the padding area)
        # must be added before patch-split processing to ensure padding is removed before merging results
        if args.get('transform_padding_region_detections'):
            processed_image_iter = PaddingPostProcesser(processed_image_iter,
                                                             max_border_distance=args.get('transform_padding_region_detections_max_border_dist'))

        # add post-processer for handling patch-split dataset (will merge them back to original images)
        processed_image_iter = PatchSplitPostProcesser(processed_image_iter,
                                                            enabled=not args.get('split_merging_disabled'),
                                                            predictions_merge_thr=args.get('split_merging_threshold_for_predictions', 0.01),)

        return dataset_it, centerdir_groundtruth_op, processed_image_iter

    def compile_evaluation_list(self):
        args = self.args

        center_checkpoint_name = args.get('center_checkpoint_name_list')
        if center_checkpoint_name is None:
            center_checkpoint_name = [args.get('center_checkpoint_name')]
        if center_checkpoint_name[0] is None:
            center_checkpoint_name = ['']

        evaluation_lists_per_center_model = {}

        for center_model_name in center_checkpoint_name:

            #########################################################################################################
            ## PREPARE EVALUATION CONFIG/ARGS

            args_eval = args['eval']
            import itertools, functools

            ##########################################################################################
            # Scoring function based on provided list of scores that we want to use

            # index should match what is returned as scores in predictions (after x,y locations)
            scoring_index = {'center': 0}

            # scoring function that extracts requested scores and multiply them to get the final score
            # scoring_fn = lambda scores: np.multiply([scores[:,scoring_index[t]] for t in args_eval['final_score_combination']])

            # function that creates new dictionaries by combinting every value in dictionary if value is a list of values
            def combitorial_args_fn(X):
                # convert to list of values
                X_vals = [[val] if type(val) not in [list, tuple] else val for val in X.values()]
                # apply product to a list of list of values
                for vals in itertools.product(*X_vals):
                    yield dict(zip(X.keys(), vals))

            # function that creates final score by multiplying specific scores (i.e. multiplying columns based on score_types names)
            def _scoring_fn(scores, score_types):
                selected_scores = [scores[:, scoring_index[t]] for t in score_types]
                return np.multiply(*selected_scores) if len(selected_scores) > 1 else selected_scores[0]

            # function that thresholds predictions based on selected columns and provided thr (as key-value pair in score_thr)
            def _scoring_thrs_fn(scores, score_thr_dict):
                selected_scores = [scores[:, scoring_index[t]] > thr for t, thr in score_thr_dict.items() if
                                   thr is not None]
                return np.multiply(*selected_scores) if len(selected_scores) > 1 else selected_scores[0]

            ##########################################################################################
            # Create all evaluation classes based on combination of thresholds and eval arguments
            evaluation_lists = []

            score_combination_and_thr_list = args_eval.get('score_combination_and_thr')

            if type(score_combination_and_thr_list) not in [list, tuple]:
                score_combination_and_thr_list = [score_combination_and_thr_list]

            # iterate over different scoring types
            for score_combination_and_thr in score_combination_and_thr_list:
                # create scoring function based on which scores are requested
                scoring_fn = functools.partial(_scoring_fn, score_types=list(score_combination_and_thr.keys()))
                # iterate over different combination of thresholds for scores that are used
                for scoring_thrs in combitorial_args_fn(score_combination_and_thr):
                    # create thresholding function based on which thresholds are requested for each score
                    scoring_thrs_fn = functools.partial(_scoring_thrs_fn, score_thr_dict=scoring_thrs)
                    # iterate over all final_score_thr that have been requested
                    for final_score_thr in args_eval.get('score_thr_final', -np.inf):
                        scoring_str = "+".join(scoring_thrs.keys())
                        scoring_thr_str = "-".join(["%s=%.2f" % (t, thr) for t, thr in scoring_thrs.items() if thr is not None])
                        exp_name = "%s-final_score_thr=%.2f-%s" % (scoring_str,final_score_thr,scoring_thr_str)

                        if args_eval.get('centers_global_minimization_for_patch_merge') is not None:
                            ##########################################################################################
                            ## Evaluation class for counting metric using data merging (for Acacia and Oilpalm datasets)
                            ## and best-fit detection minimization
                            center_eval = [PatchMergeCenterGlobalMinimizationEval(exp_name=exp_name, **args)
                                                for args in combitorial_args_fn(args_eval['centers_global_minimization_for_patch_merge'])]

                        elif args_eval.get('centers_global_minimization') is not None:
                            ##########################################################################################
                            ## Evaluation class for counting metric based on best-fit detection minimization
                            center_eval = [CenterGlobalMinimizationEval(exp_name=exp_name, **args)
                                                for args in combitorial_args_fn(args_eval['centers_global_minimization'])]
                        else:
                            ##########################################################################################
                            ## Evaluation class for centers using only AP
                            center_eval = [CenterEvaluation()]


                        evaluation_lists.append(dict(
                            scoring_fn=scoring_fn,
                            scoring_thrs_fn=scoring_thrs_fn,
                            final_score_thr=final_score_thr,
                            center_eval=center_eval)
                        )

            # Check if evaluations already exists and return empty list if needed
            if args.get('skip_if_exists') and args['save_dir'] is not None:
                if self._check_if_results_exist(evaluation_lists, self._get_save_dir(center_model_name)):
                    evaluation_lists = []

            evaluation_lists_per_center_model[center_model_name] = evaluation_lists

        return evaluation_lists_per_center_model

    def _get_checkpoint_timestamp(self):
        get_date_fn = lambda p: os.path.getmtime(p) if os.path.exists(p) else 0
        args = self.args

        if args.get('cross_validation_folds'):
            mod_date = max([get_date_fn(args['checkpoint_path'].replace("fold=XYZ","fold=%d" % fold)) for fold in args.get('cross_validation_folds')])
        else:
            mod_date = get_date_fn(args['checkpoint_path'])

        return mod_date
    
    def _check_if_results_exist(self, evaluation_lists, save_dir):
        # consider results invalid if they are older than checkpoint modification time
        checkpoint_time = self._get_checkpoint_timestamp()

        # checkpoint does not exist so just return false
        if checkpoint_time == 0:
            return False

        c_eval_exists, i_eval_exists = [], []
        for eval_args in evaluation_lists:
            C = [c_eval.get_results_timestamp(save_dir) >= checkpoint_time for c_eval in eval_args['center_eval']]

            c_eval_exists.append(all(C))

        return all(c_eval_exists) and all(i_eval_exists)

    def _get_save_dir(self, center_model_name):
        MARKER = self.args['center_checkpoint_name'] if 'center_checkpoint_name' in self.args else '##CENTER_MODEL_NAME##'

        if MARKER in self.args['save_dir']:
            return self.args['save_dir'].replace(MARKER, center_model_name)
        else:
            return self.args['save_dir']

    #########################################################################################################
    ## MAIN RUN FUNCTION
    def run(self, evaluation_lists_per_center_model):
        args = self.args

        with torch.no_grad():

            #########################################################################################################
            ## PROCESS EACH IMAGE and DO EVALUATION
            for im_index,(sample,result) in enumerate(self.processed_image_iter(self.dataset_it, self.centerdir_groundtruth_op)):

                im = sample['image']
                im_name = sample['im_name']
                base, _ = os.path.splitext(os.path.basename(im_name))

                instances = sample['instance']
                ignore_flags = sample['ignore']
                gt_centers_dict = sample['center_dict']

                output = result['output']
                predictions_ = result['predictions']
                pred_heatmap = result['pred_heatmap']
                center_model_name = result['center_model_name']

                # get difficult mask based on ignore flags (VALUE of 8 == difficult flag)
                difficult = (ignore_flags & 8 > 0).squeeze() if ignore_flags is not None else torch.zeros_like(instances)

                all_scores = predictions_[:, 2:] if len(predictions_) > 0 else []

                assert center_model_name in evaluation_lists_per_center_model

                # evaluate for different combination of scoring, thresholding and other eval arguments
                for eval_args in evaluation_lists_per_center_model[center_model_name]:
                    scoring_fn = eval_args['scoring_fn']
                    scoring_thrs_fn = eval_args['scoring_thrs_fn']
                    final_score_thr = eval_args['final_score_thr']
                    center_eval = eval_args['center_eval']

                    if len(all_scores) > 0:
                        # 1. apply scoring function
                        predictions_score = scoring_fn(all_scores)

                        # 2. filter based on specific scoring thresholds
                        selected_pred_idx = np.where((predictions_score > final_score_thr) * scoring_thrs_fn(all_scores))[0]
                        predictions = predictions_[selected_pred_idx,:]
                        predictions_score = predictions_score[selected_pred_idx]

                    else:
                        predictions = []
                        predictions_score = []

                    gt_missed, pred_missed, pred_gt_match_by_center = False, False, []

                    # 3. do evaluation for every evaluator in center_eval

                    # collected metrics for per-center evaluation
                    for c_eval in center_eval:
                        center_eval_res = c_eval.add_image_prediction(im_name, im_index, im.shape[-2:],
                                                                      predictions, predictions_score,
                                                                      instances, gt_centers_dict, difficult)

                        if isinstance(c_eval,CenterGlobalMinimizationEval) and len(c_eval.metrics['mae']) > 0:
                            base, _ = os.path.splitext(os.path.basename(im_name))
                            base = 'mae=%02d_fn=%02d_fp=%02d_%s' % (c_eval.metrics['mae'][-1], c_eval.metrics['FN'][-1], c_eval.metrics['FP'][-1], base)

                        gt_missed, pred_missed, pred_gt_match_by_center = center_eval_res

                    if args['display'] is True or \
                            type(args['display']) is str and args['display'].lower() == 'all' or \
                            type(args['display']) is str and args['display'].lower() == 'error_gt' and gt_missed or \
                            type(args['display']) is str and args['display'].lower() == 'error' and (gt_missed or pred_missed):

                        visualize_to_folder = os.path.join(self._get_save_dir(center_model_name), c_eval.exp_name)
                        os.makedirs(visualize_to_folder,exist_ok=True)

                        gt_centers = np.array([gt_centers_dict[k] for k in sorted(gt_centers_dict.keys())])
                        is_difficult_gt = np.array([difficult[np.clip(int(c[0]),0,difficult.shape[0]-1),
                                                              np.clip(int(c[1]),0,difficult.shape[1]-1),].item() != 0 for c in gt_centers])

                        if len(predictions) > 0:
                            plot_predictions = np.concatenate((np.array(predictions)[:,:2],
                                                               pred_gt_match_by_center[:,:1]), axis=1)
                        else:
                            plot_predictions = []


                        self.visualizer.display_centerdir_predictions(im[0], output[0], pred_heatmap[0], gt_centers, is_difficult_gt,
                                                                      plot_predictions, base, visualize_to_folder,
                                                                      autoadjust_figure_size=args.get('autoadjust_figure_size'))

            ########################################################################################################
            # FINALLY, OUTPUT RESULTS TO DISPLAY and FILE

            if 'eval' not in args or args['eval']:
                for center_model_name, evaluation_lists in evaluation_lists_per_center_model.items():
                    save_dir = self._get_save_dir(center_model_name)

                    for eval_args in evaluation_lists:
                        center_eval = eval_args['center_eval']

                        ########################################################################################################
                        ## Evaluation based on center point only
                        for c_eval in center_eval:
                            c_eval.calc_and_display_final_metrics(save_dir=save_dir)


def main():
    from config import get_config_args

    args = get_config_args(dataset=os.environ.get('DATASET'), type='test')

    eval = Evaluator(args)

    # get list of all evaluations that will need to be performed (based on different combinations of thresholds etc)
    evaluation_lists = eval.compile_evaluation_list()

    # continue with initialization and running all eval, unless evaluation_lists is empty
    if any([len(e) > 0 for e in evaluation_lists.values()]):
        # initialize after checking for valid list of evaluations
        eval.initialize()
        # finally run all evaluations
        eval.run(evaluation_lists)
    else:
        print('Skipping due to already existing output')

if __name__ == "__main__":
    main()
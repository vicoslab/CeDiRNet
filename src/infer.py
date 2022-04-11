#!/usr/bin/python
import ast
import os

from matplotlib import pyplot as plt

import numpy as np

import torch

from tqdm import tqdm
from datasets import get_raw_dataset
from models import get_model, get_center_model
from utils.utils import Visualizer

class CeDiRNetInfer:
    def __init__(self, args):
        if args.get('display') and not args.get('display_to_file_only') or True:
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
        dataset = get_raw_dataset('image_folder', dataset_opts=dict(root_dir=args['input_folder'],
                                                                    pattern=args['img_pattern']))

        self.dataset_it = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0,
            pin_memory=True if args['cuda'] else False)

        ###################################################################################################
        # load model
        self.model = get_model(args['model']['name'], args['model']['kwargs'])
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        self.center_model = get_center_model(args['center_model']['name'], args['center_model']['kwargs'])
        self.center_model = torch.nn.DataParallel(self.center_model).to(self.device)

        ###################################################################################################
        # load snapshot
        if os.path.exists(args['checkpoint_path']):
            print('Loading from "%s"' % args['checkpoint_path'])
            state = torch.load(args['checkpoint_path'])
            if 'model_state_dict' in state:
                if 'module.model.segmentation_head.2.weight' in state['model_state_dict']:
                    checkpoint_input_weights = state['model_state_dict']['module.model.segmentation_head.2.weight']
                    checkpoint_input_bias = state['model_state_dict']['module.model.segmentation_head.2.bias']
                    model_output_weights = self.model.module.model.segmentation_head[2].weight
                    if checkpoint_input_weights.shape != model_output_weights.shape:
                        state['model_state_dict']['module.model.segmentation_head.2.weight'] = checkpoint_input_weights[:2, :, :, :]
                        state['model_state_dict']['module.model.segmentation_head.2.bias'] = checkpoint_input_bias[:2]
                        print('WARNING: #####################################################################################################')
                        print('WARNING: regression output shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                        print('WARNING: #####################################################################################################')

                self.model.load_state_dict(state['model_state_dict'], strict=True)
            if not args.get('center_checkpoint_path') and 'center_model_state_dict' in state:
                self.center_model.load_state_dict(state['center_model_state_dict'], strict=False)
        else:
            raise Exception('checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

        if 'center_checkpoint_path' in args:
            if os.path.exists(args['center_checkpoint_path']):
                print('Loading center model from "%s"' % args['center_checkpoint_path'])
                state = torch.load(args['center_checkpoint_path'])
                if 'center_model_state_dict' in state:
                    if 'module.instance_center_estimator.conv_start.0.weight' in state['center_model_state_dict']:
                        checkpoint_input_weights = state['center_model_state_dict']['module.instance_center_estimator.conv_start.0.weight']
                        center_input_weights = self.center_model.module.instance_center_estimator.conv_start[0].weight
                        if checkpoint_input_weights.shape != center_input_weights.shape:
                            state['center_model_state_dict'][
                                'module.instance_center_estimator.conv_start.0.weight'] = checkpoint_input_weights[:, :2, :, :]

                            print('WARNING: #####################################################################################################')
                            print('WARNING: center input shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                            print('WARNING: #####################################################################################################')

                    self.center_model.load_state_dict(state['center_model_state_dict'], strict=False)
            else:
                raise Exception('checkpoint_path {} does not exist!'.format(args['center_checkpoint_path']))

        ###################################################################################################
        # Visualizer
        self.visualizer = Visualizer(('image', 'centers'), to_file_only=args.get('display_to_file_only'))

    #########################################################################################################
    ## MAIN RUN FUNCTION
    def run(self):
        args = self.args

        with torch.no_grad():

            assert self.model is not None
            self.model.eval()

            assert self.center_model is not None
            self.center_model.eval()

            im_image = 0
            #########################################################################################################
            ## PROCESS EACH IMAGE
            for sample_ in tqdm(self.dataset_it):

                im = sample_['image']

                output_batch_ = self.model(im)

                center_output = self.center_model(output_batch_, detect_centers=True, **sample_)

                output_batch, center_pred, center_heatmap = [center_output[k] for k in ['output', 'center_pred', 'center_heatmap']]

                sample_keys = sample_.keys()

                for batch_i in range(min(self.dataset_it.batch_size, len(sample_['im_name']))):
                    im_image += 1
                    output = output_batch[batch_i:batch_i + 1]

                    sample = {k: sample_[k][batch_i:batch_i + 1] for k in sample_keys}

                    im_name = sample['im_name'][0]
                    base, _ = os.path.splitext(os.path.basename(im_name))

                    # extract prediction heatmap and sorted prediction list
                    pred_heatmap = torch.relu(center_heatmap[batch_i].unsqueeze(0).unsqueeze(0))
                    predictions = center_pred[batch_i][center_pred[batch_i, :, 0] == 1][:, 1:].cpu().numpy()

                    idx = np.argsort(predictions[:, -1])
                    predictions = predictions[idx[::-1], :]

                    final_score_thr = args['score_thr_final']

                    if len(predictions) > 0:
                        predictions_score = predictions[:,-1]

                        # filter based on specific scoring thresholds
                        selected_pred_idx = np.where((predictions_score > final_score_thr))[0]
                        predictions = predictions[selected_pred_idx, :]
                        predictions_score = predictions_score[selected_pred_idx]
                    else:
                        predictions = []
                        predictions_score = []

                    if args['display'] is True:

                        visualize_to_folder = os.path.join(self.args['save_dir'])
                        os.makedirs(visualize_to_folder, exist_ok=True)

                        gt_centers = []
                        is_difficult_gt = []

                        if len(predictions) > 0:
                            plot_predictions = np.concatenate((np.array(predictions)[:, :2],
                                                               np.zeros((len(predictions),1))), axis=1)
                        else:
                            plot_predictions = []

                        self.visualizer.display_centerdir_predictions(im[0], output[0], pred_heatmap[0], gt_centers,
                                                                      is_difficult_gt,
                                                                      plot_predictions, base, visualize_to_folder,
                                                                      autoadjust_figure_size=args.get('autoadjust_figure_size'))

def parse_args():
    import argparse, json

    parser = argparse.ArgumentParser(description='Process a folder of images with CeDiRNet.')
    parser.add_argument('--input_folder', type=str, help='path to folder with input images')
    parser.add_argument('--img_pattern', type=str, help='pattern for input images')
    parser.add_argument('--output_folder', type=str, help='path to output folder')
    parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--model', type=str, help='path to model checkpoint file')
    parser.add_argument('--localization_model', type=str, default=None, help='(optional) path to localization model checkpoint file (will override one from model)')

    cmd_args = parser.parse_args()

    with open(cmd_args.config,'r') as f:
        args = json.load(f)

    args['save_dir'] = cmd_args.output_folder
    args['checkpoint_path'] = cmd_args.model
    if cmd_args.localization_model is not None:
        args['center_checkpoint_path'] = cmd_args.localization_model
    args['input_folder'] = cmd_args.input_folder
    args['img_pattern'] = cmd_args.img_pattern

    def convert_booleans(data ):
        if isinstance(data, (dict, list)):
            for k, v in (data.items() if isinstance(data, dict) else enumerate(data)):
                if isinstance(v,str) and v.lower() in ['false','true','yes','no']:
                    data[k] = v.lower() in ['true','yes']
                convert_booleans(v)

    convert_booleans(args)

    return args

def main():
    args = parse_args()

    eval = CeDiRNetInfer(args)
    eval.initialize()
    eval.run()

if __name__ == "__main__":
    main()
import copy
import os
import torchvision
if 'InterpolationMode' in dir(torchvision.transforms):
    from torchvision.transforms import InterpolationMode
else:
    from PIL import Image as InterpolationMode


import torch
from utils import transforms as my_transforms

SORGHUM_DIR=os.environ.get('SORGHUM_DIR')
CARPK_DIR=os.environ.get('CARPK_DIR')
TREE_COUNTING_DIR=os.environ.get('TREE_COUNTING_DIR')

OUTPUT_DIR=os.environ.get('OUTPUT_DIR',default='../exp')

model_dir = os.path.join(OUTPUT_DIR,'{args[dataset][name]}',
                         'adam_lr{args[train_settings][model][lr]}_loss=l1-batch_size={args[train_settings][train_dataset][batch_size]}_{args[train_settings][num_gpus]}xGPU_epoch{args[train_settings][n_epochs]}',
                         '{args[pretrained_center_name]}_with_hardsample_batch={args[train_settings][train_dataset][hard_samples_size]}_weight_decay={args[train_settings][model][weight_decay]}',
                          '2eps_distance_px={args[train_settings][train_dataset][kwargs][fixed_bbox_size]}',
                          )
args = dict(
    cuda=True,
    # DISABLE cudnn benchmark since we are using different input sizes that would cause unnecessary benchmark calls and be almost 10x slower
    cudnn_benchmark=False,

    display=False,#'error',
    display_to_file_only=True,
    autoadjust_figure_size=True,

    eval_epoch='',
    eval=dict(
        # available score types ['center']
        score_combination_and_thr=[
            {'center': [0.1,0.01,0.05,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.94,0.99],},
        ],
        score_thr_final=[0.01],
        centers_global_minimization=dict(
            display_best_threshold=False,
            tau_thr=[5,15,20,30,40] # 5 == used in paper "Locating Objects Without Bounding Boxes" by Ribera et al.
        ),
    ),

    # for point-supervision we need to set HIGH threshold for merging detections from different patches to avoid suppressing correct detections
    split_merging_threshold_for_predictions=0.5,

    save_dir=os.path.join(model_dir,'{args[dataset][kwargs][type]}_results{args[eval_epoch]}-{args[center_checkpoint_name]}--thr{args[center_model][kwargs][local_max_thr]}'),
    checkpoint_path=os.path.join(model_dir,'checkpoint{args[eval_epoch]}.pth'),

    pretrained_center_name='generic_localization',

    center_checkpoint_name='generic_localization',
    center_checkpoint_path=None,

    dataset={
        'name': 'sorghum',
        'kwargs': {
            'normalize': False,
            'root_dir': SORGHUM_DIR,
            'type': 'test',
            'fixed_bbox_size': 20,
            'resize_factor': 1.0,
            'transform': my_transforms.get_transform([
                {
                    'name': 'Padding',
                    'opts': {
                        'keys': ('image', 'instance', 'label', 'ignore'), 'keys_bbox': ('center',),
                        'pad_to_size_factor': 32
                    }
                },
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label', 'ignore'),
                        'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
            'MAX_NUM_CENTERS':2*1024,
        },
        'centerdir_gt_opts': dict(
            use_cached_backbone_output=False,
        ),

        'batch_size': 1,
        'workers': 0
    },
    model = {
        'name': 'fpn',
        'kwargs': {
            'backbone': 'resnet101',
            'num_classes': 2,
            'use_custom_fpn': True,
            'fpn_args': {
                'decoder_segmentation_head_channels':64,
            },
        }
    },

    center_model=dict(
        name='CenterDirectionLocalization',
        kwargs=dict(
            # thresholds for conv2d processing
            local_max_thr=0.1,
            use_learnable_nn=True,
            learnable_nn_args=dict(
                return_sigmoid=False,
                inner_ch=16,
                inner_kernel=3,
                dilations=[1, 4, 8, 12],
            ),
        ),

    ),

    # settings from train config needed for automated path construction
    train_settings=dict(
        num_gpus=8,
        train_dataset=dict(
            kwargs=dict(
                fixed_bbox_size=30,
                remove_out_of_bounds_centers=True,
            ),
            batch_size=128,
            hard_samples_size=0,
        ),
        model=dict(
            lr=1e-4,
            weight_decay=0,
        ),
        n_epochs=50,
    )
)



def get_args():
    return copy.deepcopy(args)

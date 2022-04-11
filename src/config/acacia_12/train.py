import copy
import os

import torch
from utils import transforms as my_transforms

import torchvision
if 'InterpolationMode' in dir(torchvision.transforms):
    from torchvision.transforms import InterpolationMode
else:
    from PIL import Image as InterpolationMode

SORGHUM_DIR=os.environ.get('SORGHUM_DIR')
CARPK_DIR=os.environ.get('CARPK_DIR')
TREE_COUNTING_DIR=os.environ.get('TREE_COUNTING_DIR')


OUTPUT_DIR=os.environ.get('OUTPUT_DIR',default='../exp')

args = dict(

    cuda=True,
    display=True,
    display_it=10,

    num_gpus=8,

    save=True,
    save_interval=10,

    save_dir=os.path.join(OUTPUT_DIR,'{args[train_dataset][name]}', 'fold={args[train_dataset][kwargs][fold]}',
                          'adam_lr{args[model][lr]}_loss=l1-batch_size={args[train_dataset][batch_size]}_{args[num_gpus]}xGPU_epoch{args[n_epochs]}',
                          '{args[pretrained_center_name]}_with_hardsample_batch={args[train_dataset][hard_samples_size]}_weight_decay={args[model][weight_decay]}',
                          '2eps_distance_px={args[train_dataset][kwargs][fixed_bbox_size]}',
                          ),

    pretrained_model_path = None,
    resume_path = None,

    pretrained_center_name = 'generic_localization',
    pretrained_center_model_path = os.path.join(OUTPUT_DIR,"learnable-center-dilated-kernels-batch_size=768-16xGPU_epoch200/hard_negs=384-dataset_size=5000/loss=l1-lr=0.0001-w_cent=1-w_fg_cent=50/checkpoint.pth"),

    train_dataset = {
        'name': 'acacia_12',
        'kwargs': {
            'normalize': False,
            'root_dir': TREE_COUNTING_DIR,
            'type': 'train',
            'fold': 0, 'num_folds': 3,
            'fold_split_axis_rotate': 0, 'fold_split_axis': 'i', 'PATCH_SIZE': 512,
            'fixed_bbox_size': 30, #dataset does not have mask so we need to use this
            'resize_factor': 1.0,
            'remove_out_of_bounds_centers': True,
            'transform': my_transforms.get_transform([
                {
                    'name': 'RandomHorizontalFlip',
                    'opts': {
                        'keys': ('image', 'instance', 'label', 'ignore'), 'keys_bbox': ('center',),
                        'p': 0.5,
                    }
                },
                {
                    'name': 'RandomVerticalFlip',
                    'opts': {
                        'keys': ('image', 'instance', 'label', 'ignore'), 'keys_bbox': ('center',),
                        'p': 0.5,
                    }
                },
                {
                    'name': 'ColorJitter',
                    'opts': {
                        'keys': ('image',), 'p': 0.5,
                        'saturation': 0.2, 'hue': 0.2, 'brightness': 0.2, 'contrast':0.2
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
        'batch_size': 32,
        'hard_samples_size': 0,
        'hard_samples_selected_min_percent':0.1,
        'workers': 8,
        'shuffle': True,
    }, 

    model = dict(
        name='fpn',
        kwargs= {
            'backbone': 'resnet101',
            'num_classes': 2,
            'use_custom_fpn':True,
            'fpn_args': {
                'decoder_segmentation_head_channels':64,
            },
            'init_decoder_gain': 0.1
        },
        optimizer = 'Adam',
        lr = 1e-4,
        weight_decay = 0,

    ),
    center_model=dict(
        name='CenterDirectionLocalization',
        kwargs=dict(
            # thresholds for conv2d processing
            local_max_thr=0.1,
            use_learnable_nn=True,
            learnable_nn_args=dict(
                inner_ch=16,
                inner_kernel=3,
                dilations=[1, 4, 8, 12],
                freeze_learning=True,
            ),
            augmentation=False, # cannot use during training of center directions since it will interfere with it
        ),
        optimizer='Adam',
        lr=0,
        weight_decay=0,
    ),

    # --------
    n_epochs=200,

    # loss options
    loss_type='CenterDirectionLoss',
    loss_opts={
        'regression_loss': 'l1',
        'localization_loss': 'l1',

        'enable_localization_loss': False,
        'enable_direction_loss': True,
},
    loss_w={
        'w_cos': 1,
        'w_sin': 1,
        'w_cent': 0.1,
    },

)

# Original scheduler used by SpatialEmbedding method
args['lambda_scheduler_fn']=lambda _args: (lambda epoch: pow((1-((epoch)/_args['n_epochs'])), 0.9))
#args['lambda_scheduler_fn']=lambda _args: (lambda epoch: 1.0) # disabled

args['model']['lambda_scheduler_fn'] = args['lambda_scheduler_fn']
args['center_model']['lambda_scheduler_fn'] = lambda _args: (lambda epoch: pow((1-((epoch)/_args['n_epochs'])), 0.9) if epoch > 1 else 0)


def get_args():
    return copy.deepcopy(args)

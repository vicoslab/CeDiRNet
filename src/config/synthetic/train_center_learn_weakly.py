import copy
import os

import torchvision
if 'InterpolationMode' in dir(torchvision.transforms):
    from torchvision.transforms import InterpolationMode
else:
    from PIL import Image as InterpolationMode

import torch
from utils import transforms as my_transforms

OUTPUT_DIR=os.environ.get('OUTPUT_DIR',default='../exp')

args = dict(

    cuda=True,
    display=True,
    display_it=10,

    num_gpus=16,

    save=True,

    save_dir=os.path.join(OUTPUT_DIR,'synthetic',
                          'learnable-center-dilated-kernels-batch_size={args[train_dataset][batch_size]}-{args[num_gpus]}xGPU_epoch{args[n_epochs]}',
                          'hard_negs={args[train_dataset][hard_samples_size]}-dataset_size={args[train_dataset][kwargs][length]}',
                          'loss={args[loss_name]}-lr={args[center_model][lr]}-w_cent={args[loss_w][w_cent]}-w_fg_cent={args[loss_w][w_fg_cent]}',
                          ),
    resume_path=None,

    pretrained_model_path=None,

    pretrained_center_model_name='',
    pretrained_center_model_path=None,

    train_dataset = {
        'name': 'syn',
        'kwargs': {
            'normalize': False,
            'length': 5000,
            'image_size': (512,512),
            'max_instance_overlap':0,
            'allow_out_of_bounds':False,
            'num_instances':[5.,50],
            'instance_size_relative':[0.025,0.10],
            'instance_aspect_ratio':[0.5,2],
            'instance_rotation':[-180,180],
            'siblings_probability': 0.25,
            'siblings_count':[2,5] ,
            'siblings_resize_probability': 0.25,
            'siblings_resize_range': [0.5,2],
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
            'MAX_NUM_CENTERS':1024,
        },
        'centerdir_gt_opts': dict(
            center_extend_px=3,
            center_gt_blur=2,

            use_cached_backbone_output=True,
            add_synthetic_output=True,
        ),
        'batch_size': 768,
        'hard_samples_size': 384,
        'workers': 8,
    },

    model=dict(
        name='fpn',
        kwargs={
            'backbone': 'resnet101',
            'num_classes': 2,
            'use_custom_fpn':True,
            'fpn_args': {
                'decoder_segmentation_head_channels':64,
            },
            'init_decoder_gain': 0.1
        },
        disabled=True,
        optimizer='Adam',
        lr=0,
        weight_decay=0,
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
            ),
            augmentation=True,
            augmentation_kwargs=dict(
                occlusion_probability=0.75,
                occlusion_type='circle',
                occlusion_distance_type='larger', #random
                occlusion_center_jitter_probability=0.5,
                occlusion_center_jitter_relative_size=0.4,
                gaussian_noise_probability=0.25,
                gaussian_noise_blur_sigma=3,
                gaussian_noise_std_polar=[0.1,2.0],
                gaussian_noise_std_mask=[0.1,2.0]
            ),
        ),
        optimizer='Adam',
        lr=1e-4,
        weight_decay=0,
    ),

    #---
    n_epochs=200,

    # loss options
    loss_type='CenterDirectionLoss',
    loss_opts={
        'regression_loss': 'l1',
        'localization_loss': 'l1',

        'enable_localization_loss': True,
        'enable_direction_loss': False,
    },
    loss_name='l1',
    loss_w={
        'w_cos': 0,
        'w_sin': 0,
        'w_cent': 1,
        'w_fg_cent': 50,
        'w_bg_cent': 1,
    },

)
# Original scheduler used by SpatialEmbedding method
args['lambda_scheduler']=lambda epoch: pow((1-((epoch)/args['n_epochs'])), 0.9),
#args['lambda_scheduler']=lambda epoch: 1.0, # disabled

args['model']['lambda_scheduler'] = args['lambda_scheduler']
args['center_model']['lambda_scheduler'] = args['lambda_scheduler']

def get_args():
    return copy.deepcopy(args)

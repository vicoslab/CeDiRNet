import os
import shutil
import time, datetime

from matplotlib import pyplot as plt
from tqdm import tqdm

import numpy as np

import json
import torch
import torch.distributed
import torch.multiprocessing

from datasets import get_dataset
from models import get_model, get_center_model
from criterions import get_criterion
from utils.utils import DistributedRandomSampler, HardExamplesBatchSampler
from utils.utils import AverageMeter, Logger, Visualizer, distributed_sync_dict
from utils import transforms as my_transforms

from config import get_config_args

from utils.evaluation.center_global_min import CenterGlobalMinimizationEval

class Trainer:
    def __init__(self, local_rank, rank_offset, world_size, args, use_distributed_data_parallel=True):
        self.args = args
        self.world_size = world_size
        self.world_rank = rank_offset + local_rank
        self.local_rank = local_rank

        self.use_distributed_data_parallel = use_distributed_data_parallel and world_size > 1

        if args['save'] and self.world_rank == 0:
            if not os.path.exists(args['save_dir']):
                os.makedirs(args['save_dir'])

            # save parameters
            with open(os.path.join(args['save_dir'],'params.json'), 'w') as file:
                file.write(json.dumps(args, indent=4, sort_keys=True,default=lambda o: '<not serializable>'))

        if args['display']:
            plt.ion()
        else:
            plt.ioff()
            plt.switch_backend("agg")

    def initialize_data_parallel(self, init_method=None):
        ###################################################################################################
        # set device
        if self.use_distributed_data_parallel and init_method is not None:
            self.use_distributed_data_parallel = True

            self.device = torch.device("cuda:%d" % self.local_rank)

            # if not master, then wait at least 5 sec to give master a chance for starting up first
            if self.world_rank != 0:
                time.sleep(10)

            # initialize the process group
            torch.distributed.init_process_group("nccl", init_method=init_method, timeout=datetime.timedelta(hours=1),
                                                 rank=self.world_rank, world_size=self.world_size)

            print('Waiting for all nodes (ready from rank=%d/%d)' % (self.world_rank, self.world_size))
            torch.distributed.barrier()
        else:
            self.use_distributed_data_parallel = False
            self.device = torch.device("cuda:0" if self.args['cuda'] else "cpu")

    def cleanup(self):
        if self.use_distributed_data_parallel:
            torch.distributed.destroy_process_group()

    def _to_data_parallel(self, X, **kwargs):
        if self.use_distributed_data_parallel:
            X = torch.nn.parallel.DistributedDataParallel(X.to(self.device), device_ids=[self.local_rank], find_unused_parameters=True, **kwargs)
        else:
            X = torch.nn.DataParallel(X.to(self.device), device_ids=[self.device], **kwargs)
        return X

    def _synchronize_dict(self, array):
        if self.use_distributed_data_parallel:
            array = distributed_sync_dict(array, self.world_size, self.world_rank, self.device)
        return array

    def initialize(self):
        args = self.args
        device = self.device

        ###################################################################################################
        # train dataloader
        dataset_workers = args['train_dataset']['workers'] if 'workers' in args['train_dataset'] else 0
        dataset_batch = args['train_dataset']['batch_size'] if 'batch_size' in args['train_dataset'] else 1
        dataset_shuffle = args['train_dataset']['shuffle'] if 'shuffle' in args['train_dataset'] else True
        dataset_hard_sample_size = args['train_dataset'].get('hard_samples_size')

        # in distributed settings we need to manually reduce batch size
        if self.use_distributed_data_parallel:
            dataset_batch = dataset_batch // self.world_size
            dataset_workers = 0 # ignore workers request since already using separate processes for each GPU
            if dataset_hard_sample_size:
                dataset_hard_sample_size = dataset_hard_sample_size // self.world_size

        train_dataset, centerdir_groundtruth_op = get_dataset(args['train_dataset']['name'], args['train_dataset']['kwargs'],
                                                              args['train_dataset'].get('centerdir_gt_opts'))

        if centerdir_groundtruth_op is not None:
            centerdir_groundtruth_op = self._to_data_parallel(centerdir_groundtruth_op)

        # prepare hard-examples sampler for dataset
        if dataset_shuffle:
            if self.use_distributed_data_parallel:
                default_sampler = DistributedRandomSampler(train_dataset, device=self.device)
            else:
                default_sampler = torch.utils.data.RandomSampler(train_dataset)
        else:
            default_sampler = torch.utils.data.SequentialSampler(train_dataset)

        batch_sampler = HardExamplesBatchSampler(train_dataset,
                                                 default_sampler,
                                                 batch_size=dataset_batch,
                                                 hard_sample_size=dataset_hard_sample_size,
                                                 drop_last=True, hard_samples_selected_min_percent=args['train_dataset'].get('hard_samples_selected_min_percent'),
                                                 device=self.device, world_size=self.world_size, rank=self.world_rank,
                                                 is_distributed=self.use_distributed_data_parallel)

        train_dataset_it = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=batch_sampler, num_workers=dataset_workers, pin_memory=True if args['cuda'] else False)

        ###################################################################################################
        # set model
        model = get_model(args['model']['name'], args['model']['kwargs'])
        model.init_output()
        model = self._to_data_parallel(model, dim=0)

        # set center prediction head model
        center_model = get_center_model(args['center_model']['name'], args['center_model']['kwargs'])
        center_model.init_output()
        center_model = self._to_data_parallel(center_model, dim=0)

        # set criterion
        criterion = get_criterion(args.get('loss_type'), args.get('loss_opts'), model.module, center_model.module)
        criterion = self._to_data_parallel(criterion, dim=0)

        def get_optimizer(model_, args_):
            if args_ is None or args_.get('disabled'):
                return None, None
            if 'optimizer' not in args_ or args_['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(model_.parameters(),lr=args_['lr'], weight_decay=args_['weight_decay'])
            elif args_['optimizer'] == 'SGD':
                optimizer = torch.optim.SGD(model_.parameters(),lr=args_['lr'], momentum=args_['momentum'],
                                            weight_decay=args_['weight_decay'])
            # use custom lambda_scheduler_fn function that can pass args if available
            lr_lambda = args_['lambda_scheduler_fn'](args) if 'lambda_scheduler_fn' in args_ else args_['lambda_scheduler']
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

            return optimizer, scheduler

        # set optimizer for model and for center model
        optimizer, scheduler = get_optimizer(model, args['model'])
        center_optimizer, center_scheduler = get_optimizer(center_model, args['center_model'])

        # Visualizer
        visualizer = Visualizer(('image', 'centers', 'centerdir_gt', 'conv_centers'))

        # Logger
        self.logger = Logger(('train',), 'loss')

        # resume
        self.start_epoch = 0
        if args['resume_path'] is not None and os.path.exists(args['resume_path']):
            print('Resuming model from {}'.format(args['resume_path']))
            state = torch.load(args['resume_path'])
            self.start_epoch = state['epoch'] + 1
            if 'model_state_dict' in state: model.load_state_dict(state['model_state_dict'], strict=True)
            if 'optim_state_dict' in state and optimizer: optimizer.load_state_dict(state['optim_state_dict'])
            if 'center_model_state_dict' in state: center_model.load_state_dict(state['center_model_state_dict'], strict=True)
            if 'center_optim_state_dict' in state and center_optimizer: center_optimizer.load_state_dict(state['center_optim_state_dict'])
            self.logger.data = state['logger_data']

        if args.get('pretrained_model_path') is not None and os.path.exists(args['pretrained_model_path']):
            print('Loading pre-trained model from {}'.format(args['pretrained_model_path']))
            state = torch.load(args['pretrained_model_path'])
            if 'model_state_dict' in state: model.load_state_dict(state['model_state_dict'], strict=True)
            if 'center_model_state_dict' in state: center_model.load_state_dict(state['center_model_state_dict'], strict=False)

        if args.get('pretrained_center_model_path') is not None and os.path.exists(args['pretrained_center_model_path']):
            print('Loading pre-trained center model from {}'.format(args['pretrained_center_model_path']))
            state = torch.load(args['pretrained_center_model_path'])

            INPUT_WEIGHTS_KEY = 'module.instance_center_estimator.conv_start.0.weight'
            if INPUT_WEIGHTS_KEY in state['center_model_state_dict']:
                checkpoint_input_weights = state['center_model_state_dict'][INPUT_WEIGHTS_KEY]
                center_input_weights = center_model.module.instance_center_estimator.conv_start[0].weight
                if checkpoint_input_weights.shape != center_input_weights.shape:
                    state['center_model_state_dict'][INPUT_WEIGHTS_KEY] = checkpoint_input_weights[:, :2, :, :]

                    print('WARNING: #####################################################################################################')
                    print('WARNING: center input shape mismatch - will load weights for only the first two channels, is this correct ?!!!')
                    print('WARNING: #####################################################################################################')

            center_model.load_state_dict(state['center_model_state_dict'], strict=False)

        denormalize_args = None

        # get prepare values/functions needed for display
        if 'transform' in args['train_dataset']['kwargs']:
            transforms = args['train_dataset']['kwargs']['transform'].transforms
            denormalize_args = [(t.mean[t.keys == 'image'], t.std[t.keys == 'image'])
                                for t in transforms if type(t) == my_transforms.Normalize and 'image' in t.keys]
            denormalize_args = denormalize_args[0] if len(denormalize_args) > 0 else None

        if 'learnable_center_loss' in args['loss_opts'] and args['loss_opts']['learnable_center_loss'] == 'cross-entropy':
            center_conv_resp_fn = lambda x: torch.sigmoid(x)
        else:
            center_conv_resp_fn = lambda x: x

        self.device = device
        self.batch_sampler, self.train_dataset_it, self.dataset_batch = batch_sampler, train_dataset_it, dataset_batch
        self.model, self.center_model = model, center_model
        self.scheduler, self.center_scheduler = scheduler, center_scheduler
        self.optimizer, self.center_optimizer = optimizer, center_optimizer
        self.criterion, self.centerdir_groundtruth_op  = criterion, centerdir_groundtruth_op

        self.center_conv_resp_fn = center_conv_resp_fn
        self.visualizer, self.denormalize_args = visualizer, denormalize_args

    def print(self, *kargs, **kwargs):
        if self.world_rank == 0:
            print(*kargs, **kwargs)

    def _get_difficulty_scores(self, loss_total, center_pred, gt_centers_dict, difficult, ignore):

        # losses to find difficult/hard examples
        losses_for_hard_neg = torch.zeros((len(loss_total),), dtype=torch.float, device=loss_total.device)

        FN_b, FP_b = 0.0, 0.0
        for b in range(len(loss_total)):
            # calc FP and FN
            if center_pred is not None:
                center_eval = CenterGlobalMinimizationEval()
                valid_pred = center_pred[b, center_pred[b, :, 0] != 0, :]
                valid_pred = valid_pred[ignore[b, 0, valid_pred[:, 2].long(), valid_pred[:, 1].long()] == 0, :] if ignore is not None else valid_pred
                center_eval.add_image_prediction(None, None, None, valid_pred[:, 1:3].cpu().numpy(), None, None, gt_centers_dict[b], difficult[b])
                FP, FN = center_eval.metrics['FP'][0], center_eval.metrics['FN'][0]
            else:
                FP, FN = 0, 0

            # multiply loss with number of FP and FN for hard neg
            losses_for_hard_neg[b] = loss_total[b].sum() * (FP + 2 * FN + 1) ** 2

            FN_b += FN
            FP_b += FP

        return losses_for_hard_neg, FN_b, FP_b

    def train(self, epoch):
        args = self.args

        device = self.device
        batch_sampler, train_dataset_it, dataset_batch = self.batch_sampler, self.train_dataset_it, self.dataset_batch
        model, center_model = self.model, self.center_model
        optimizer, center_optimizer = self.optimizer, self.center_optimizer
        criterion, centerdir_groundtruth_op = self.criterion, self.centerdir_groundtruth_op

        # sum only over channels
        reduction_dim = (1,2,3)

        # put model into training mode
        model.train()
        if optimizer and args['center_model']['lr'] > 0:
            center_model.train()
        else:
            center_model.eval()

        # define meters
        loss_meter = AverageMeter()

        if optimizer:
            for param_group in optimizer.param_groups:
                self.print('learning rate (model): {}'.format(param_group['lr']))
        if center_optimizer:
            for param_group in center_optimizer.param_groups:
                self.print('learning rate (center_model): {}'.format(param_group['lr']))

        iter=epoch*len(train_dataset_it)

        current_iter_sample_loss = {}

        epoch_iter = tqdm(train_dataset_it) if self.world_rank == 0 or True else train_dataset_it
        for i, sample in enumerate(epoch_iter):

            # call centerdir_groundtruth_op first which will create any missing centerdir_groundtruth (using GPU) and add synthetic output
            # and also update model function if requested to use cached outputs (or requested to save them)
            if centerdir_groundtruth_op is not None:
                sample = centerdir_groundtruth_op(sample, torch.arange(0, dataset_batch).int())
                model = centerdir_groundtruth_op.module.insert_cached_model(model, sample)

            im = sample['image']
            instances = sample['instance'].squeeze(dim=1)
            class_labels = sample['label'].squeeze(dim=1)
            ignore = sample.get('ignore')
            centerdir_gt = sample.get('centerdir_groundtruth')
            centers = sample.get('center')

            # use any centers provided by dataset if exists
            if centers is not None:
                gt_centers_dict = [{id: centers[b, id, [1, 0]].cpu().numpy()
                                   for id in range(centers[b].shape[0]) if centers[b, id, 0] > 0 and centers[b, id, 1] > 0 and id in torch.unique(instances[b])}
                                        for b in range(len(centers))]
            else:
                gt_centers_dict = [{id.item(): torch.nonzero(instances[b].squeeze() == id).float().mean(dim=0).cpu().numpy()
                                   for id in torch.unique(instances[b]) if id > 0}
                                        for b in range(len(im))]

            loss_ignore = None
            if ignore is not None:
                # treat any type of ignore objects (truncated, border, etc) as ignore during training
                # (i.e., ignore loss and any groundtruth objects at those pixels)
                loss_ignore = ignore > 0

                gt_centers_dict = [{k: c for k, c in gt_centers_dict[b].items()
                                    if loss_ignore[b, 0][instances[b] == k].min() == 0}
                                            for b in range(len(im))]

            # get difficult mask based on ignore flags (VALUE of 8 == difficult flag and VALUE of 2 == truncated flag )
            difficult = (((ignore & 8) | (ignore & 2)) > 0).squeeze(dim=1) if ignore is not None else torch.zeros_like(instances)

            # retrieve and set random seed for hard examples from previous epoch
            # (will be returned as None if sample does not exist or is not hard-sample)
            sample['seed'] = batch_sampler.retrieve_hard_sample_storage_batch(sample['index'],'seed')

            output = model(im)

            # call center prediction model
            center_output = center_model(output, **sample)
            output, center_pred, center_heatmap = [center_output[k] for k in ['output', 'center_pred', 'center_heatmap']]

            # get losses
            losses = criterion(output, instances, class_labels,
                               centerdir_responses=(center_pred, center_heatmap), centerdir_gt=centerdir_gt, ignore_mask=loss_ignore,
                               reduction_dims=reduction_dim, **args['loss_w'])

            # since each GPU will have only portion of data it will not use correct batch size for averaging - do correction for this here
            if self.world_size > 1 and not self.use_distributed_data_parallel:
                losses = [l/float(self.world_size) for l in losses]

            loss_total, loss_direction_total, loss_centers, loss_sin, loss_cos = losses

            # find difficult/hard samples
            samples_difficulty_score, FN, FP = self._get_difficulty_scores(loss_total, center_pred, gt_centers_dict,
                                                                           difficult, ignore)

            # save loss for each sample during this epoch
            for b in range(len(loss_total)):
                current_iter_sample_loss[sample['index'][b].item()] = loss_total[b].sum().item()

            # pass losses to hard-samples batch sampler
            batch_sampler.update_sample_loss_batch(sample, samples_difficulty_score, index_key='index', storage_keys=['seed'])

            loss = losses[0].sum()

            if optimizer: optimizer.zero_grad()
            if center_optimizer: center_optimizer.zero_grad()

            loss.backward()

            if optimizer: optimizer.step()
            if center_optimizer: center_optimizer.step()

            if self.world_rank == 0:
                epoch_iter.set_postfix(dict(epoch=epoch, loss=loss.item(), FP=FP, FN=FN))

                if args['display'] and i % args['display_it'] == 0:

                    center_conv_resp = self.center_conv_resp_fn(center_heatmap) if center_heatmap is not None else None
                    self.visualizer.display_centerdir_training(im, output, center_conv_resp, centerdir_gt, gt_centers_dict,
                                                               difficult, 0, device, self.denormalize_args)

            loss_meter.update(loss.item())

            iter+=1

        current_iter_sample_loss = self._synchronize_dict(current_iter_sample_loss)

        if epoch == args['n_epochs']:
            self.print('end')

        return np.array(list(current_iter_sample_loss.values())).mean() * dataset_batch
    def save_checkpoint(self, state, name='checkpoint.pth'):
        args = self.args

        print('=> saving checkpoint')
        file_name = os.path.join(args['save_dir'], name)
        torch.save(state, file_name)
        if state['epoch'] % args.get('save_interval',10) == 0:
            shutil.copyfile(file_name, os.path.join(args['save_dir'], 'checkpoint_%03d.pth' % state['epoch']))

    def should_skip_training(self):
        last_interval = self.args['n_epochs'] - self.args.get('save_interval',10)
        last_checkpoint = os.path.join(self.args['save_dir'], 'checkpoint_%03d.pth' % last_interval)

        return self.args.get('skip_if_exists') and os.path.exists(last_checkpoint)

    def run(self):
        args = self.args

        for epoch in range(self.start_epoch, args['n_epochs']):

            if self.world_rank == 0: print('Starting epoch {}'.format(epoch))
            if self.scheduler: self.scheduler.step(epoch)
            if self.center_scheduler: self.center_scheduler.step(epoch)

            train_loss = self.train(epoch)

            if self.world_rank == 0:
                print('===> train loss: {:.2f}'.format(train_loss))

                self.logger.add('train', train_loss)

                self.logger.plot(save=args['save'], save_dir=args['save_dir'])

                if args['save'] and (epoch % args.get('save_interval',10) == 0 or epoch + 1 == args['n_epochs']):
                    state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict() if self.model is not None else None,
                        'optim_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                        'center_model_state_dict': self.center_model.state_dict() if self.center_model is not None else None,
                        'center_optim_state_dict': self.center_optimizer.state_dict() if self.center_optimizer is not None else None,
                        'logger_data': self.logger.data,
                    }
                    self.save_checkpoint(state)

def main(local_rank, rank_offset, world_size, init_method=None):

    args = get_config_args(dataset=os.environ.get('DATASET'), type='train')

    trainer = Trainer(local_rank, rank_offset, world_size, args, use_distributed_data_parallel=init_method is not None)

    if trainer.should_skip_training():
        print('Skipping due to already existing checkpoints (and requested to skip if exists) !!')
        return

    trainer.initialize_data_parallel(init_method)

    trainer.initialize()
    trainer.run()

    trainer.cleanup()

import torch.multiprocessing as mp

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    world_size = int(os.environ.get('WORLD_SIZE',default=n_gpus))
    rank_offset = int(os.environ.get('RANK_OFFSET',default=0))

    if world_size <= 1:
        main(0, 0, n_gpus)
    else:
        args = get_config_args(dataset=os.environ.get('DATASET'), type='train')
        init_methods = 'env://'

        spawn = None
        try:
            print("spawning %d new processes" % n_gpus)
            spawn = mp.spawn(main,
                             args=(rank_offset,world_size,init_methods),
                             nprocs=n_gpus,
                             join=False)
            while not spawn.join():
                pass
        except KeyboardInterrupt:
            if spawn is not None:
                for pid in spawn.pids():
                    os.system("kill %s" % pid)
            torch.distributed.destroy_process_group()

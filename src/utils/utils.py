import collections
import os
import threading
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import cv2
import torch

class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x/y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class)/len(self.avg_per_class)

class Visualizer:

    def __init__(self, keys, to_file_only=False):
        self.wins = {k:None for k in keys}

        self.to_file_only = to_file_only

    def display(self, image, key, title=None, denormalize_args=None, autoadjust_figure_size=False, **kwargs):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1
    
        if self.wins[key] is None:
            n_cols = int(np.maximum(1,np.ceil(np.sqrt(n_images))))
            n_rows = int(np.maximum(1,np.ceil(n_images/n_cols)))
            self.wins[key] = plt.subplots(ncols=n_cols,nrows=n_rows)
    
        fig, ax = self.wins[key]

        if isinstance(ax, collections.Iterable):
            ax = ax.reshape(-1)
            n_axes = len(ax)
        else:
            n_axes= 1
        assert n_images <= n_axes
    
        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
            ax.imshow(self.prepare_img(image, denormalize_args), **kwargs)
            max_size = image.shape[-2:]
        else:
            max_size = 0,0
            for i,ax_i in enumerate(ax):
                ax_i.cla()
                ax_i.set_axis_off()
                if i < n_images:
                    ax_i.imshow(self.prepare_img(image[i], denormalize_args), **kwargs)
                    max_size = max(max_size[0], image[i].shape[-2]), max(max_size[1], image[i].shape[-1])

        fig.subplots_adjust(top=1.0, bottom=0.00, left=0.01, right=0.99, wspace=0.01,hspace=0.01)

        if title is not None:
            fig.suptitle(title)

        if autoadjust_figure_size:
            if min(max_size) > 2*1024:
                f = (2*1024.0)/min(max_size)
                max_size = max_size[0]*f, max_size[1]*f

            n_cols, n_rows = fig.axes[0].get_subplotspec().get_gridspec().get_geometry()
            fig.set_size_inches(n_rows*max_size[1]/100, n_cols*max_size[0]/100)

        if True:
            plt.draw()
            self.mypause(0.001)

        return fig,ax

    def display_opencv(self, image, key, save=None, title=None, denormalize_args=None, autoadjust_figure_size=False,
                       plot_fn = None, image_colormap=None, **kwargs):
        def prepare_img_cv(im, colormap=cv2.COLORMAP_PARULA):
            im = self.prepare_img(im, denormalize_args)

            if len(im.shape) == 3 and im.shape[-1] == 3:
                # convert RGB to BGR
                im = im[:,:,[2,1,0]]

            if len(im.shape) == 2 and im.dtype.kind != 'u':
                im = (((im-im.min()) / (im.max() - im.min())) * 255).astype(np.uint8)
                im = cv2.applyColorMap(im, colormap)
            elif im.dtype.kind != 'u':
                im = (im * 255).astype(np.uint8)

            im = np.ascontiguousarray(im, dtype=np.uint8)

            if plot_fn is not None:
                im = plot_fn(im)

            return im

        if not isinstance(image, (list, tuple)):
            image = [image]

        n_images = len(image)

        n_cols = int(np.maximum(1, np.ceil(np.sqrt(n_images))))
        n_rows = int(np.maximum(1, np.ceil(n_images / n_cols)))

        if image_colormap is None:
            image_colormap = [cv2.COLORMAP_PARULA]*n_images
        elif not isinstance(image_colormap, (list, tuple)):
            image_colormap = [image_colormap]

        # prepare images
        I = [prepare_img_cv(I,cmap) for I,cmap in zip(image,image_colormap)]

        # convert to grid with n_cols and n_rows
        if len(I) < n_cols*n_rows:
            I += [np.ones_like(I[0])*255] * (n_cols*n_rows - len(I))

        I = np.concatenate([np.concatenate(I[i*n_cols:(i+1)*n_cols], axis=1) for i in range(n_rows)],axis=0)

        return I



    @staticmethod
    def prepare_img(image, denormalize_args=None):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()
            if denormalize_args is not None:
                denorm_mean,denorm_std = denormalize_args

                image = (image * denorm_std.numpy().reshape(-1,1,1)) +  denorm_mean.numpy().reshape(-1,1,1)


        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return


    def display_centerdir_predictions(self, im, output, pred_heatmap, gt_list, is_difficult_gt,
                                      pred_list, base, save_dir=None, autoadjust_figure_size=False):
        if self.to_file_only:
            self._display_centerdir_predictions_opencv(im, output, pred_heatmap,
                                                       gt_list, is_difficult_gt, pred_list, base, save_dir,
                                                       autoadjust_figure_size)
            return
        def plot_prediction_markers(ax_, **plot_args):
            if len(pred_list) > 0:
                pred_list_true = pred_list[pred_list[:, 2] > 0, :]
                pred_list_false = pred_list[pred_list[:, 2] <= 0, :]

                ax_.plot(pred_list_true[:, 0], pred_list_true[:, 1], 'gx', **plot_args)
                ax_.plot(pred_list_false[:, 0], pred_list_false[:, 1], 'rx', **plot_args)


        fig_img, ax = self.display(im.cpu(), 'image', force_draw=False, autoadjust_figure_size=autoadjust_figure_size)

        plot_prediction_markers(ax, markersize=10, markeredgewidth=2)

        if len(gt_list[is_difficult_gt == 0]) > 0:
            ax.plot(gt_list[is_difficult_gt == 0, 1], gt_list[is_difficult_gt==0, 0], 'g.',
                    markersize=5, markeredgewidth=0.2, markerfacecolor=(0,1,0,1), markeredgecolor=(0, 0, 0, 1))
        if len(gt_list[is_difficult_gt != 0]) > 0:
            ax.plot(gt_list[is_difficult_gt!=0, 1], gt_list[is_difficult_gt != 0, 0], 'y.',
                    markersize=5, markeredgewidth=0.2, markerfacecolor=(1,1,0,1), markeredgecolor=(0, 0, 0, 1))

        fig_centers, ax = self.display([(output[1]).detach().cpu(),
                                        (output[0]).detach().cpu(),
                                        pred_heatmap.detach().cpu()], 'centers', force_draw=False,
                                       autoadjust_figure_size=autoadjust_figure_size)
        for ax_i in ax:
            plot_prediction_markers(ax_i, markersize=4, markeredgewidth=1)

        if save_dir is not None:
            fig_img.savefig(os.path.join(save_dir, '%s_0.img.png' % base))
            fig_centers.savefig(os.path.join(save_dir, '%s_1.centers.png' % base))


    def _display_centerdir_predictions_opencv(self, im, output, pred_heatmap, gt_list, is_difficult_gt,
                                              pred_list, base, save_dir=None, autoadjust_figure_size=False):
        from functools import partial

        def plot_prediction_markers(img, gt=False, predictions_args=dict(), bbox_args=dict()):
            predictions_args_ = dict(markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            predictions_args_.update(predictions_args)
            bbox_args_ = dict(thickness=3)
            bbox_args_.update(bbox_args)
            if len(pred_list) > 0:
                pred_list_true = pred_list[pred_list[:, 2] > 0, :]
                pred_list_false = pred_list[pred_list[:, 2] <= 0, :]

                for p in pred_list_true: cv2.drawMarker(img, (int(p[0]),int(p[1])), color=(0,255,0), **predictions_args_)
                for p in pred_list_false: cv2.drawMarker(img, (int(p[0]),int(p[1])), color=(0,0,255), **predictions_args_)
            if gt:
                for i,p in enumerate(gt_list):
                    cv2.circle(img, (int(p[1]), int(p[0])), radius=3, color=(0, 255, 0) if is_difficult_gt[i] == 0 else (0, 255, 255), thickness=-1)
                    cv2.circle(img, (int(p[1]), int(p[0])), radius=3, color=(0, 0, 0), thickness=1)

            return img

        fig_img = self.display_opencv(im.cpu(), 'image', autoadjust_figure_size=autoadjust_figure_size,
                                      plot_fn=partial(plot_prediction_markers, gt=True))

        fig_centers = self.display_opencv([(output[1]).detach().cpu(),
                                           (output[0]).detach().cpu(),
                                           torch.atan2(output[0],output[1]).detach().cpu(),
                                           pred_heatmap.detach().cpu()], 'centers',
                                          autoadjust_figure_size=autoadjust_figure_size,
                                          plot_fn=partial(plot_prediction_markers, predictions_args=dict(thickness=1)),
                                          image_colormap=[cv2.COLORMAP_PARULA,cv2.COLORMAP_PARULA,cv2.COLORMAP_HSV,cv2.COLORMAP_PARULA])

        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '%s_0.img.png' % base), fig_img)
            cv2.imwrite(os.path.join(save_dir, '%s_1.centers.png' % base), fig_centers)

    def display_centerdir_training(self, im, output, center_conv_resp=None, centerdir_gt=None, gt_centers_dict=None, gt_difficult=None,
                                   plot_batch_i=0, device=None, denormalize_args=None):
        with torch.no_grad():
            gt_list = np.array([c for k, c in gt_centers_dict[plot_batch_i].items()]) if gt_centers_dict is not None else []
            is_difficult_gt = np.array([False] * len(gt_list))
            if gt_difficult is not None:
                gt_difficult = gt_difficult[plot_batch_i]
                is_difficult_gt = np.array([gt_difficult[np.clip(int(c[0]),0,gt_difficult.shape[0]-1),
                                                         np.clip(int(c[1]),0,gt_difficult.shape[1]-1)].item() != 0 for c in gt_list])
            def plot_gt(ax_):
                if type(ax_) not in [list, tuple, np.ndarray]:
                    ax_ = [ax_]
                gt_list_easy, gt_list_hard = gt_list[is_difficult_gt == 0], gt_list[is_difficult_gt != 0]

                for a in ax_:
                    if len(gt_list_easy) > 0:
                        a.plot(gt_list_easy[:, 1], gt_list_easy[:, 0], 'g.', markersize=5, markeredgewidth=0.2,
                               markerfacecolor=(0, 1, 0, 1), markeredgecolor=(0, 0, 0, 1))
                    if len(gt_list_hard) > 0:
                        a.plot(gt_list_hard[:, 1], gt_list_hard[:, 0], 'y.', markersize=5, markeredgewidth=0.2,
                               markerfacecolor=(1, 1, 0, 1), markeredgecolor=(0, 0, 0, 1))


            _,ax = self.display(im[plot_batch_i].cpu(), 'image', denormalize_args=denormalize_args)
            plot_gt(ax)

            out_viz = [(output[plot_batch_i, 1]).detach().cpu(),
                       (output[plot_batch_i, 0]).detach().cpu()]
            if centerdir_gt is not None and len(centerdir_gt) > 0:
                gt_R, gt_theta, gt_sin_th, gt_cos_th = centerdir_gt[0][:, 0], centerdir_gt[0][:, 1], centerdir_gt[0][:, 2], \
                                                       centerdir_gt[0][:, 3]

                out_viz += [torch.abs(output[plot_batch_i, 1] - gt_cos_th[plot_batch_i, 0].to(device)).detach().cpu(),
                            torch.abs(output[plot_batch_i, 0] - gt_sin_th[plot_batch_i, 0].to(device)).detach().cpu()]
            else:
                out_viz += [(output[plot_batch_i, 3 + i]).detach().cpu() for i in
                            range(len(output[plot_batch_i, :]) - 4)]

            _, ax = self.display(out_viz, 'centers')
            plot_gt(ax)

            if center_conv_resp is not None:
                conv_centers = center_conv_resp[plot_batch_i].detach().cpu()
                if centerdir_gt is not None and len(centerdir_gt) > 0:
                    gt_center_mask = centerdir_gt[0][plot_batch_i, 5]
                    conv_centers = [conv_centers,
                                    torch.abs(center_conv_resp[plot_batch_i] - gt_center_mask.to(device)).detach().cpu()]

                _, ax = self.display(conv_centers, 'conv_centers')
                plot_gt(ax)

            if centerdir_gt is not None and len(centerdir_gt) > 0:
                _, ax = self.display((centerdir_gt[0][plot_batch_i, 0].cpu(),
                                    centerdir_gt[0][plot_batch_i, 1].cpu(),
                                    centerdir_gt[0][plot_batch_i, 2].cpu(),
                                    centerdir_gt[0][plot_batch_i, 3].cpu(),), 'centerdir_gt')
                plot_gt(ax)
class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

    def add(self, key, value):
        assert key in self.data, "Key not in data"
        self.data[key].append(value)

import scipy
import torch.nn as nn

class GaussianLayer(nn.Module):
    def __init__(self, num_channels=1, sigma=3):
        super(GaussianLayer, self).__init__()

        self.sigma = sigma
        self.kernel_size = int(2 * np.ceil(3*self.sigma - 0.5) + 1)

        self.conv = nn.Conv2d(num_channels, num_channels, self.kernel_size, stride=1,
                              padding=self.kernel_size//2, bias=None, groups=num_channels)


        self.weights_init()
    def forward(self, x):
        return self.conv(x)

    def weights_init(self):
        n = np.zeros((self.kernel_size,self.kernel_size))
        n[self.kernel_size//2,self.kernel_size//2] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=self.sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))


from torch.utils.data.sampler import Sampler, RandomSampler, WeightedRandomSampler, BatchSampler

class _WeightedRandomSampler(Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [0, 0, 0, 1, 0]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    def __init__(self, weights, num_samples, replacement=True):
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples

from datasets import LockableSeedRandomAccess

import pickle

import torch.distributed as dist

class DistributedRandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, device=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.device = device

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            iter_order = torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).to(self.device)
        else:
            iter_order = torch.randperm(n).to(self.device)

        # ensure order is the same for all processes (use iter from rank-0)
        dist.broadcast(iter_order,0)

        return iter(iter_order.tolist())

    def __len__(self):
        return self.num_samples


class DistributedSubsetRandomSampler(Sampler):
    def __init__(self, indices, device=None):
        self.indices = indices
        self.device = device

    def __iter__(self):
        iter_order = torch.randperm(len(self.indices)).to(self.device)

        # ensure order is the same for all processes (use iter from rank-0)
        dist.broadcast(iter_order,0)

        return (self.indices[i.item()] for i in iter_order)

    def __len__(self):
        return len(self.indices)

def distributed_sync_dict(array, world_size, rank, device, MAX_LENGTH=10*2**20): # default MAX_LENGTH = 10MB
    def _pack_data(_array):
        data = pickle.dumps(_array)
        data_length = int(len(data))
        data = data_length.to_bytes(4, "big") + data
        assert len(data) < MAX_LENGTH
        data += bytes(MAX_LENGTH - len(data))
        data = np.frombuffer(data, dtype=np.uint8)
        assert len(data) == MAX_LENGTH
        return torch.from_numpy(data)
    def _unpack_data(_array):
        data = _array.to(torch.uint8).cpu().numpy().tobytes()
        data_length = int.from_bytes(data[:4], 'big')
        return pickle.loads(data[4:data_length+4])
    def _unpack_size(_array):
        print(_array.shape, _array[:4])
        data = _array.to(torch.uint8).cpu().numpy().tobytes()
        data_length = int.from_bytes(data[:4], 'big')
        print(data_length,data[:4])
        return data_length

    # prepare output buffer
    output_tensors = [torch.zeros(MAX_LENGTH, dtype=torch.uint8, device=device) for _ in range(world_size)]
    # pack data using pickle into input/output
    output_tensors[rank][:] = _pack_data(array)

    # sync data
    dist.all_gather(output_tensors, output_tensors[rank])

    # unpack data and merge into single dict
    return {id:val for array_tensor in output_tensors for id,val in _unpack_data(array_tensor).items()}

class HardExamplesBatchSampler(Sampler):

    def __init__(self, dataset, default_sampler, batch_size, hard_sample_size, drop_last, hard_samples_selected_min_percent=0,
                 device=None, world_size=None, rank=None, is_distributed=False):
        if not isinstance(default_sampler, Sampler):
            raise ValueError("default_sampler should be an instance of "
                             "torch.utils.data.Sampler, but got default_sampler={}"
                             .format(default_sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not (isinstance(hard_sample_size, int) or hard_sample_size is None) or \
                hard_sample_size < 0 or hard_sample_size >= batch_size :
            raise ValueError("hard_sample_size should be a positive integer value smaller than batch_size, "
                             "but got hard_sample_size={}".format(hard_sample_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.is_distributed = is_distributed and world_size > 1
        self.world_size = world_size if self.is_distributed else 1
        self.rank = rank if self.is_distributed else 0
        self.device = device

        self.dataset = dataset
        self.default_sampler = default_sampler
        if self.is_distributed:
            self.hard_sampler = DistributedSubsetRandomSampler(list(range(len(default_sampler))),device=device)
        else:
            self.hard_sampler = torch.utils.data.SubsetRandomSampler(list(range(len(default_sampler))))
        self.hard_sample_size = hard_sample_size if hard_sample_size is not None else 0
        self.hard_samples_selected_min_percent = hard_samples_selected_min_percent if hard_samples_selected_min_percent is not None else 0
        self.batch_size = batch_size
        self.drop_last = drop_last


        self.sample_losses = dict()
        self.sample_storage = dict()
        self.sample_storage_tmp = dict()

    def update_sample_loss_batch(self, gt_sample, losses, index_key='index', storage_keys=[]):
        assert index_key in gt_sample, "Index key %s is not present in gt_sample" % index_key

        indices = gt_sample[index_key]

        # convert to numpy
        indices = indices.detach().cpu().numpy() if isinstance(indices, torch.Tensor) else indices
        losses = losses.detach().cpu().numpy() if isinstance(losses, torch.Tensor) else losses

        for i,l in enumerate(losses):
            # get id of the sample (i.e. its index key)
            id = indices[i]

            # store its loss value
            self.sample_losses[id] = l
            # store any additional info required to pass along for hard examples
            # (save to temporary array which will be used for next epoch)
            self.sample_storage_tmp[id] = {k:gt_sample[k][i] for k in storage_keys}

    def retrieve_hard_sample_storage_batch(self, ids, key=None):
        # convert to numpy
        ids = ids.detach().cpu().numpy() if isinstance(ids, torch.Tensor) else ids
        # return matching sample_storage value for hard examples (i.e. for first N samples, where N=self.hard_sample_size)
        return [self.sample_storage[id][key] if n < self.hard_sample_size and id in self.sample_storage else None for n,id in enumerate(ids)]

    def _synchronize_dict(self, array):
        return distributed_sync_dict(array, self.world_size, self.rank, self.device)

    def _recompute_hard_samples_list(self):
        if self.is_distributed:
            self.sample_losses = self._synchronize_dict(self.sample_losses)
        if len(self.sample_losses) > 0:
            k = np.array(list(self.sample_losses.keys()))
            v = np.array([self.sample_losses[i] for i in k])
            v = (v - v.mean()) / v.std()
            hard_ids = list(k)
            for std_thr in [2, 1, 0.5, 0]:
                new_hard_ids = list(k[v > std_thr])
                if len(new_hard_ids) > len(v)*self.hard_samples_selected_min_percent:
                    hard_ids = new_hard_ids
                    break
            self.hard_sampler.indices = hard_ids if len(hard_ids) > 0 else list(k)
            if self.rank == 0:
                print('Number of hard samples present: %d/%d' % (len(hard_ids), len(self.sample_losses)))

        if isinstance(self.dataset,LockableSeedRandomAccess):
            # lock seeds for hard samples BUT not for the whole dataset i.e. 90% of the whole dataset
            # (otherwise this will fully lock seeds for all samples and prevent new random augmentation of samples)
            self.dataset.lock_samples_seed(self.hard_sampler.indices if len(self.hard_sampler.indices) < len(self.sample_losses)*0.9 else [])

        # update storage for next iteration
        self.sample_storage = self._synchronize_dict(self.sample_storage_tmp) if self.is_distributed else self.sample_storage_tmp
        self.sample_storage_tmp = dict()

    def __iter__(self):
        from itertools import islice
        self._recompute_hard_samples_list()
        max_index = len(self.default_sampler)
        if self.drop_last:
            total_batch_size = self.batch_size * self.world_size
            max_index = (max_index // total_batch_size) * total_batch_size

        batch = []
        hard_iter = iter(self.hard_sampler)
        self.usage_freq = {i: 0 for i in range(len(self.default_sampler))}
        for idx in islice(self.default_sampler,self.rank,max_index,self.world_size):
            batch.append(idx)
            # stop when spaces for normal samples filled
            if len(batch) == self.batch_size-self.hard_sample_size:
                # fill remaining places with hard examples
                # (does not need to be sync for distributed since sampling is random with replacement)
                while len(batch) < self.batch_size:
                    try:
                        batch.insert(0,next(hard_iter))
                    except StopIteration: # reset iter if no more samples
                        hard_iter = iter(self.hard_sampler)

                for b in batch: self.usage_freq[b] += 1
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            for b in batch: self.usage_freq[b] += 1
            yield batch

    def get_avg_sample_loss(self):
        return np.array(list(self.sample_losses.values())).mean()

    def get_sample_losses(self):
        return self.sample_losses.copy()

    def get_sample_frequency_use(self):
        return self.usage_freq.copy()

    def __len__(self):
        size_default = len(self.default_sampler)

        if self.is_distributed:
            size_default = size_default // self.world_size

        actual_batch_size = self.batch_size-self.hard_sample_size
        if self.drop_last:
            return size_default // actual_batch_size
        else:
            return (size_default + actual_batch_size - 1) // actual_batch_size



####################################################################################################################
# General class that can merge split image results - can merge tensor data (image or heatmap) and points
class ImageGridCombiner:
    class Data:
        def __init__(self, w,h):
            self.full_size = [h,w]
            self.full_data = dict()
            self.image_names = []
        def add_image_name(self, name):
            self.image_names.append(name)

        def get_image_name(self):
            org_names = [n.split("_patch")[0] + ".png" for n in self.image_names]
            # check that all images had the same original name
            org_names = np.unique(org_names)

            if len(org_names) != 1:
                raise Exception("Invalid original names found: %s" % ",".join(org_names))

            return org_names[0]

        def set_tensor2d(self, name, partial_data, roi_x, roi_y, merge_op=None):
            if name not in self.full_data:
                self.full_data[name] = torch.zeros(list(partial_data.shape[:-2]) + self.full_size, dtype=partial_data.dtype, device=partial_data.device)

            full_data_roi = self.full_data[name][..., roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]]
            partial_data_roi = partial_data[..., 0:(roi_y[1] - roi_y[0]),0:(roi_x[1] - roi_x[0])].type(full_data_roi.dtype)
            try:
                # default merge operator is to use max unless specified otherwise
                if merge_op is None:
                    merge_op = lambda Y,X: torch.where(Y.abs() < X.abs(), X, Y)

                self.full_data[name][..., roi_y[0]:roi_y[1], roi_x[0]:roi_x[1]] = merge_op(full_data_roi, partial_data_roi)
            except:
                print('error')

        def set_instance_description(self, name, partial_instance_list, partial_instance_mask, reverse_desc, x, y):
            if name not in self.full_data:
                self.full_data[name] = ([],[])

            for i,desc in partial_instance_list.items():
                i_mask = (partial_instance_mask == i).nonzero().cpu().numpy()

                self.full_data[name][0].append(desc + np.array(([y,x] if reverse_desc else [x,y]) + [0]*(len(desc)-2)))
                self.full_data[name][1].append(i_mask + np.array([y, x]))

        def get(self, data_names):
            return [self.full_data.get(n) for n in data_names]

        def get_instance_description(self, name, out_mask_tensor, overlap_thr=0):
            instance_list, instance_mask_ids = self.full_data[name]

            # clip mask ids to out_mask_tensor.shape
            instance_mask_ids = [np.array([c for c in i_mask if c[0] >= 0 and c[1] >= 0 and c[0] < out_mask_tensor.shape[0] and c[1] < out_mask_tensor.shape[1]])
                                        for i_mask in instance_mask_ids]

            instance_indexes = [set(np.ravel_multi_index((np.array(i_mask)[:, 0], np.array(i_mask)[:, 1]), dims=out_mask_tensor.shape)) if len(i_mask) > 0 else set()
                                        for i_mask in instance_mask_ids]

            retained = self._find_retained_instances(instance_indexes, overlap_thr)

            instance_list = [x for i, x in enumerate(instance_list) if retained[i]]
            instance_mask_ids = [x for i, x in enumerate(instance_mask_ids) if retained[i]]
            instance_indexes = [x for i, x in enumerate(instance_indexes) if retained[i]]

            instance_dict = {}
            instance_indexes_dict = {}

            for id, (i_center, i_mask, i_mask_indexes) in enumerate(zip(instance_list, instance_mask_ids, instance_indexes)):
                i_mask_ids = torch.from_numpy(i_mask)
                if len(i_mask_ids) > 0:
                    out_mask_tensor[(i_mask_ids[:, 0], i_mask_ids[:, 1])] = id + 1
                    instance_dict[id + 1] = i_center
                    instance_indexes_dict[id+1] = list(i_mask_indexes)

            return instance_dict, out_mask_tensor, instance_indexes_dict

        def _find_retained_instances(self, instance_indexes, overlap_thr):
            retained = np.ones(shape=len(instance_indexes), dtype=np.bool)
            for i in range(len(instance_indexes)):
                if retained[i]:
                    for j in range(i + 1, len(instance_indexes)):
                        inter_ij = len(instance_indexes[i].intersection(instance_indexes[j]))
                        iou_ratio = inter_ij / (len(instance_indexes[i]) + len(instance_indexes[j]) - inter_ij + 1e-5)
                        if iou_ratio > overlap_thr:
                            if len(instance_indexes[i]) > len(instance_indexes[j]):
                                retained[j] = False
                            else:
                                retained[i] = False
                                break
            return retained
    def __init__(self):
        self.current_index = None
        self.current_data = None

    def add_image(self, im_name, grid_index, data_map_tensor, data_map_instance_desc, custom_merge_ops={}):
        n, x, y, w, h, org_w, org_h = grid_index

        # set roi and clamp to max size
        roi_x = x, min(x + w, org_w)
        roi_y = y, min(y + h, org_h)

        finished_data = None

        if self.current_index is None or n != self.current_index:
            finished_data = self.current_data
            self.current_data = self.Data(org_w, org_h)
            self.current_index = n

        self.current_data.add_image_name(im_name)

        if n == self.current_index:
            for name,partial_data in data_map_tensor.items():
                if partial_data is not None:
                    self.current_data.set_tensor2d(name, partial_data, roi_x, roi_y, merge_op=custom_merge_ops.get(name))

            for name,partial_data in data_map_instance_desc.items():
                if partial_data is not None:
                    self.current_data.set_instance_description(name, partial_data[0], partial_data[1], partial_data[2], x,y)

        return finished_data


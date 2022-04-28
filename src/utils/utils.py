import os
import threading
import matplotlib.pyplot as plt
import numpy as np

import torch

from utils.vis import Visualizer


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


from torch.utils.data.sampler import Sampler, WeightedRandomSampler


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


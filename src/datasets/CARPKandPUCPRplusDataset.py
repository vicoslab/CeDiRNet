import glob
import os, sys
import pickle

import scipy.io

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
from skimage.segmentation import relabel_sequential


import torch
from torch.utils.data import Dataset

from .LockableSeedRandomAccess import LockableSeedRandomAccess

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CARPKandPUCPRplusDataset(Dataset, LockableSeedRandomAccess):

    class_names = ('car',)
    class_ids = (1,)

    IGNORE_FLAG = 1
    IGNORE_TRUNCATED_FLAG = 2
    IGNORE_OVERLAP_BORDER_FLAG = 4
    IGNORE_DIFFICULT_FLAG = 8

    def __init__(self, root_dir='./', type="train", db_name='CARPK', class_id=1, MAX_NUM_CENTERS=1024, BORDER_MARGIN_FOR_CENTER=0,
                 remove_out_of_bounds_centers=False, fixed_bbox_size=None, resize_factor=None,
                 transform=None, normalize=False, valid_img_names=None):

        print('%s Dataset created' % db_name)

        self.class_id = class_id

        self.transform = transform
        self.normalize = normalize

        self.resize_factor = resize_factor
        self.fixed_bbox_size = fixed_bbox_size
        self.remove_out_of_bounds_centers = remove_out_of_bounds_centers

        self.MAX_NUM_CENTERS = MAX_NUM_CENTERS
        self.BORDER_MARGIN_FOR_CENTER = BORDER_MARGIN_FOR_CENTER

        print('Loading image list and groundtruths ...')

        root_dir = os.path.join(root_dir,'%s_devkit' % db_name,'data')

        # get image and instance list
        image_list = glob.glob(os.path.join(root_dir, 'Images','*.jpg'))
        image_list += glob.glob(os.path.join(root_dir, 'Images','*.png'))
        image_list.sort()

        # parse gt list into dictionary of groundtruths
        self.gt_dict = self._load_gt_dict(os.path.join(root_dir,'Annotations'))

        # read list of train/test images
        split_img_names = np.loadtxt(os.path.join(root_dir,'ImageSets','%s.txt' % type), dtype=str)

        image_list = [l for l in image_list if len([v for v in split_img_names if v in l]) > 0]

        if valid_img_names is not None and len(valid_img_names) > 0:
            image_list = [l for l in image_list if len([v for v in valid_img_names if v in l]) > 0]

        self.image_list = image_list
        self.real_size = len(self.image_list)

        # generate a list of seeds so that we generate the same data for each index
        self.seed_list = np.random.randint(sys.maxsize, size=self.real_size)

        self.return_gt_heatmaps = True
        self.return_gt_polygon = False
        self.return_image = True

    def lock_samples_seed(self, index_list):
        # regenerate seed for indexes that are not locked
        for index in list(set(range(self.real_size)) - set(index_list)):
            self.seed_list[index] = np.random.randint(sys.maxsize)

    def __len__(self):
        return self.real_size

    def __getitem__(self, index):
        # this will load only image info but not the whole data
        image = Image.open(self.image_list[index])
        im_size = image.size

        if self.resize_factor is not None:
            im_size = int(image.size[0] * self.resize_factor), int(image.size[1] * self.resize_factor)

        # avoid loading full buffer data if image not requested
        if self.return_image:
            if self.resize_factor is not None and self.resize_factor != 1.0:
                image = image.resize(im_size, Image.BILINEAR)
        else:
            image = None

        sample = dict(image=image,
                      im_name=self.image_list[index],
                      im_size=im_size,
                      index=index)

        gt_bbox_list = self._load_gt_bbox(index)

        if self.return_gt_heatmaps:
            instance, label, ignore, center = self.decode_instance((im_size[1],im_size[0]),
                                                                   gt_bbox_list, self.class_id, self.MAX_NUM_CENTERS)

            sample.update(dict(instance=instance.convert("I"), # ensure single channel
                               label=label,
                               ignore=ignore, # encode: 1 == ignore, 2 == truncated objects
                               center=center))

        if self.return_gt_polygon:
            # convert from box to polygon
            gt_polygon_list = np.stack((gt_bbox_list[:, 0, 0], gt_bbox_list[:, 0, 1],
                                        gt_bbox_list[:, 0, 0], gt_bbox_list[:, 1, 1],
                                        gt_bbox_list[:, 1, 0], gt_bbox_list[:, 1, 1],
                                        gt_bbox_list[:, 1, 0], gt_bbox_list[:, 0, 1]),axis=1)
            gt_polygon_list = gt_polygon_list.reshape(-1,4,2)

            sample['instance_polygon'] = gt_polygon_list

        # transform
        if self.transform is not None:
            rng = np.random.default_rng(seed=self.seed_list[index])
            sample = self.transform(sample, rng)

        if self.normalize:
            sample['image'] = (np.array(sample['image']) - 128)/ 128  # normalize to [-1,1]

        # ensure instance is int16
        if 'instance' in sample:
            to_int16_img = lambda X: X.type(torch.int16) if isinstance(X,torch.Tensor) else X.convert("I;16")
            sample['instance'] = to_int16_img(sample['instance'])

        if self.remove_out_of_bounds_centers:
            # if instance has out-of-bounds center then ignore it if requested so
            out_of_bounds_ids = [id for id, c in enumerate(sample['center'])
                             # if center closer to border then this margin than mark it as truncated
                             if id > 0 and (c[0] < 0 or c[1] < 0 or
                                            c[0] >= sample['image'].shape[-1] or
                                            c[1] >= sample['image'].shape[-2])]
            for id in out_of_bounds_ids:
                sample['instance'][sample['instance'] == id] = 0
                sample['center'][id,:] = -1

        if self.transform is not None:
            # recheck for ignore regions due to changes from augmentation:
            #  - mark with ignore any centers/instances that are now outside of image size
            valid_ids = np.unique(sample['instance'])
            truncated_ids = [id for id, c in enumerate(sample['center'])
                             # if center closer to border then this margin than mark it as truncated
                             if id > 0 and id in valid_ids and (c[0] < self.BORDER_MARGIN_FOR_CENTER or
                                                                c[1] < self.BORDER_MARGIN_FOR_CENTER or
                                                                c[0] >= sample['image'].shape[-1]-self.BORDER_MARGIN_FOR_CENTER or
                                                                c[1] >= sample['image'].shape[-2]-self.BORDER_MARGIN_FOR_CENTER)]
            for id in truncated_ids:
                sample['ignore'][sample['instance'] == id] |= self.IGNORE_TRUNCATED_FLAG

        return sample

    def _load_gt_bbox(self, n):
        folder = os.path.dirname(self.image_list[n])
        filename = os.path.splitext(os.path.basename(self.image_list[n]))[0]

        # load instances from loaded csv
        return self._load_gt_list(self.gt_dict[filename], self.fixed_bbox_size, self.resize_factor)

    @classmethod
    def _load_gt_dict(cls, gt_dir):

        gt_dict = {}
        print('Loading groundtruth files')
        for gt_file in tqdm(glob.glob(os.path.join(gt_dir, '*.txt'))):
            gt = np.loadtxt(os.path.join(gt_dir,gt_file)).reshape(-1,5)
            gt_dict[os.path.basename(gt_file).replace(".txt","")] = gt[gt[:,-1] == 1,:4]

        return gt_dict

    @classmethod
    def _load_gt_list(cls, gt_bbox_list, fixed_bbox_size=None, resize_factor=None):

        if len(gt_bbox_list) > 0:

            # add space for object size and switch X,Y position
            if resize_factor is not None:
                gt_bbox_list[:, :] *= resize_factor

            if fixed_bbox_size is not None:
                gt_center_list = gt_bbox_list.reshape(-1, 2, 2).mean(axis=1)
                gt_center_list = np.concatenate((gt_center_list, np.zeros_like(gt_center_list)),axis=1)

                gt_center_list[:, 2] = fixed_bbox_size[0] if type(fixed_bbox_size) in [list,tuple] else fixed_bbox_size
                gt_center_list[:, 3] = fixed_bbox_size[1] if type(fixed_bbox_size) in [list,tuple] else fixed_bbox_size

                # convert center list to polygon list (x1,y1,x2,y2)
                gt_bbox_list = np.array([gt_center_list[:, 0]-gt_center_list[:, 2]/2, gt_center_list[:, 1]-gt_center_list[:, 3]/2,
                                         gt_center_list[:, 0]+gt_center_list[:, 2]/2, gt_center_list[:, 1]+gt_center_list[:, 3]/2]).T
        else:
            gt_bbox_list = np.zeros(shape=(0, 4))

        gt_bbox_list = gt_bbox_list.reshape(-1, 2, 2)

        return gt_bbox_list

    @classmethod
    def decode_instance(cls, pic_size, gt_bbox_list, class_id=None, MAX_NUM_CENTERS=1024):

        pic = np.zeros(pic_size,dtype=np.uint16)
        instance_map = np.zeros(pic_size, dtype=np.uint16)
        ignore_map = np.zeros(pic_size, dtype=np.uint8)

        if gt_bbox_list is not None:
            # clip groundtruth to within image
            gt_bbox_list[:, :, 0] = np.clip(gt_bbox_list[:, :, 0], 0, pic_size[1] - 1)
            gt_bbox_list[:, :, 1] = np.clip(gt_bbox_list[:, :, 1], 0, pic_size[0] - 1)

            for i, bbox in enumerate(gt_bbox_list):
                current_obj = pic[int(bbox[0][1]):int(bbox[1][1]),
                                  int(bbox[0][0]):int(bbox[1][0])]
                if current_obj.sum() > 0:
                    # overlap with existing object - assign pixels to closer one only
                    current_bboxes = np.array([bbox] + [gt_bbox_list[id-1] for id in np.unique(current_obj) if id > 0])
                    current_centers = (current_bboxes[:,0]+current_bboxes[:,1])/2
                    X,Y = np.meshgrid(np.arange(int(bbox[0][0]),int(bbox[1][0])),
                                      np.arange(int(bbox[0][1]),int(bbox[1][1])))
                    dist = np.stack([np.sqrt((X-c[0])**2+(Y-c[1])**2) for c in current_centers])
                    valid_pic = np.argmin(dist,axis=0) == 0
                    pic[int(bbox[0][1]):int(bbox[1][1]),
                        int(bbox[0][0]):int(bbox[1][0])][valid_pic] = i+1
                else:
                    pic[int(bbox[0][1]):int(bbox[1][1]),
                        int(bbox[0][0]):int(bbox[1][0])] = i+1

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(pic_size, dtype=np.uint8)

        if len(gt_bbox_list) > MAX_NUM_CENTERS:
            raise Exception("Too small MAX_NUM_CENTERS (increase it manually!!)")

        center_list = np.zeros((MAX_NUM_CENTERS, 2), dtype=np.float)

        if class_id is not None:
            mask = np.array(pic) > 0
            if mask.sum() > 0:
                ids, forward_map, _ = relabel_sequential(pic[mask])
                instance_map[mask] = ids
                class_map[mask] = 1

                # remap all centers as well using forward_map
                all_centers = (gt_bbox_list[:, 0] + gt_bbox_list[:, 1]) / 2
                for n,center in enumerate(all_centers):
                    id = forward_map[n+1]
                    if id > 0:
                        center_list[id] = center

        else:
            raise Exception()


        return Image.fromarray(instance_map), Image.fromarray(class_map), Image.fromarray(ignore_map), center_list


if __name__ == "__main__":
    import pylab as plt


    from src.utils import transforms as my_transforms

    import torchvision

    from torchvision.transforms import InterpolationMode

    transform = my_transforms.get_transform([
        {
            'name': 'Padding',
            'opts': {
                'keys': ('image', 'instance', 'label', 'ignore'), 'keys_bbox': ('center',),
                'pad_to_size_factor': 32,
                'borders': (64, 64, 64, 64),
                # 'padding_mode':'edge',
                'padding_mode': 'symmetric',
                'mark_key_with_flag': 'ignore'
            }
        },
        {
            'name': 'ToTensor',
            'opts': {
                'keys': ('image', 'instance', 'label', 'ignore'),
                'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor),
            }
        },
    ])
    db = CARPKandPUCPRplusDataset(root_dir='/storage/datasets/CARPK/', type='train', db_name='CARPK', resize_factor=0.5,
                                  fixed_bbox_size=30, MAX_NUM_CENTERS=2048, transform=transform)
    #db.return_gt_polygon = True
    shapes = []
    for item in tqdm(db):
        if item['index'] % 50 == 0:
            print('loaded index %d' % item['index'])
        shapes.append(item['image'].shape)
        if True or np.array(item['ignore']).sum() > 0:
            center = item['center']
            gt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :]

            plt.clf()

            plt.subplot(1,2,1)
            plt.imshow(item['image'].permute([1,2,0]))
            plt.plot(gt_centers[:, 0], gt_centers[:, 1], 'r.')

            plt.subplot(1, 2, 2)
            plt.imshow(item['ignore'][0])
            #plt.imshow(item['instance'][0])
            plt.plot(gt_centers[:, 0], gt_centers[:, 1], 'r.')
            plt.show(block=False)

    print("end")
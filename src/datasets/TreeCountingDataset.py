import ast
import glob
import os, sys
import pickle

from tqdm import tqdm

import numpy as np
from PIL import Image, ImageFile
from skimage.segmentation import relabel_sequential

import torch
from torch.utils.data import Dataset

from .LockableSeedRandomAccess import LockableSeedRandomAccess

ImageFile.LOAD_TRUNCATED_IMAGES = True

class TreeCountingDataset(Dataset, LockableSeedRandomAccess):

    class_names = ('tree',)
    class_ids = (1,)

    IGNORE_FLAG = 1
    IGNORE_TRUNCATED_FLAG = 2
    IGNORE_OVERLAP_BORDER_FLAG = 4
    IGNORE_DIFFICULT_FLAG = 8

    def __init__(self, root_dir='./', name='Acacia_06', type="train", class_id=1, fold=0, num_folds=3,
                 fold_split_axis='i', fold_split_axis_rotate=0, PATCH_SIZE=512, VALIDATION_REGION_SIZE=768, MAX_NUM_CENTERS=1024,
                 BORDER_MARGIN_FOR_CENTER=0, remove_out_of_bounds_centers=False, fixed_bbox_size=None, resize_factor=None,
                 transform=None, normalize=False, valid_img_names=None):

        assert fixed_bbox_size, "TreeCounting requires fixed_bbox_size parameter"

        print('TreeCounting Dataset created')

        self.class_id = class_id

        self.transform = transform
        self.normalize = normalize

        self.resize_factor = resize_factor
        self.fixed_bbox_size = fixed_bbox_size

        self.MAX_NUM_CENTERS = MAX_NUM_CENTERS
        self.BORDER_MARGIN_FOR_CENTER = BORDER_MARGIN_FOR_CENTER

        self.remove_out_of_bounds_centers = remove_out_of_bounds_centers

        root_dir = os.path.join(root_dir, name)

        assert os.path.exists(root_dir)

        print('Loading image list and groundtruths ...')

        # get image and instance list
        image_list = glob.glob(os.path.join(root_dir, '*.jpg'))
        image_list.sort()

        # parse image patch location from its name
        patch_locations = {}
        for im_name in image_list:
            # get i and j location in global image space
            loc_dict = self.get_image_global_location(im_name)

            if len(loc_dict) > 0:
                patch_locations[im_name] = loc_dict

        # rotate patch location into axis requested for train/test splitting
        if fold_split_axis_rotate != 0:
            max_i_loc = max([loc['i'] for loc in patch_locations.values()])
            max_j_loc = max([loc['j'] for loc in patch_locations.values()])

            rotate_i = lambda loc, th: (loc['i'] - max_i_loc / 2) * np.cos(th) - (loc['j'] - max_j_loc / 2) * np.sin(th)
            rotate_j = lambda loc, th: (loc['i'] - max_i_loc / 2) * np.sin(th) + (loc['j'] - max_j_loc / 2) * np.cos(th)

            new_patch_locations = {k:dict(i=rotate_i(loc, -fold_split_axis_rotate * (np.pi/180)),
                                          j=rotate_j(loc, -fold_split_axis_rotate * (np.pi/180)))
                                    for k,loc in patch_locations.items()}

            if False:
                import pylab as plt
                plt.ion()
                x = np.array([loc['i'] for loc in patch_locations.values()])
                y = np.array([loc['j'] for loc in patch_locations.values()])
                plt.plot(x, y, '.b'); plt.axis('equal')

                X = np.array([loc['i'] for loc in new_patch_locations.values()])
                Y = np.array([loc['j'] for loc in new_patch_locations.values()])
                plt.figure()
                plt.plot(X, Y, '.b'); plt.axis('equal')

            patch_locations = new_patch_locations

        # prepare fn that will return location based on fold split axis
        get_split_axis_loc = lambda loc: loc[fold_split_axis]
        get_split_axis_ortogonal_loc = lambda loc: loc['i' if fold_split_axis == 'j' else 'j']

        # find min/max to align location values
        axis_loc_list = [get_split_axis_loc(loc) for loc in patch_locations.values()]
        max_loc, min_loc = max(axis_loc_list), min(axis_loc_list)

        test_fold_idx_size = (max_loc + PATCH_SIZE - min_loc) / num_folds

        orto_axis_loc_list = [get_split_axis_ortogonal_loc(loc) for loc in patch_locations.values()]
        max_orto_loc, min_orto_loc = max(orto_axis_loc_list), min(orto_axis_loc_list)

        max_orto_loc - min_orto_loc

        # prepare function for checking into which split the image falls, e.g for num_folds=3:
        #
        # fold=0: --------------------------   fold=1:  --------------------------  fold=2: --------------------------
        #         |  TEST  | train | train |           | train |  TEST  | train |           | train | train |  TEST  |
        #         -------------------------            --------------------------           --------------------------

        dist_to_test_border_fn = lambda X: min(map(np.abs,(test_fold_idx_size * fold - X, test_fold_idx_size * fold - (X + PATCH_SIZE), \
                                                           test_fold_idx_size * (fold + 1) - X, test_fold_idx_size * (fold + 1) - (X + PATCH_SIZE))))

        dist_to_test_fold_fn = lambda X: min(abs(np.floor(X / test_fold_idx_size) - fold),
                                             abs(np.floor((X + PATCH_SIZE) / test_fold_idx_size) - fold))

        is_test_fn = lambda X: dist_to_test_fold_fn(X) == 0
        is_train_fn = lambda X: not is_test_fn(X) and dist_to_test_border_fn(X) >= PATCH_SIZE/2
        is_val_fn = lambda Y: np.abs((max_orto_loc - min_orto_loc)/2 - Y) <= VALIDATION_REGION_SIZE

        # select images based on train/test for selected fold
        if type == 'test':
            image_list = [l for l in image_list if is_test_fn(get_split_axis_loc(patch_locations[l]) - min_loc)]
        elif type == 'train':
            image_list = [l for l in image_list if is_train_fn(get_split_axis_loc(patch_locations[l]) - min_loc) and not is_val_fn(get_split_axis_ortogonal_loc(patch_locations[l]) - min_orto_loc)]
        elif type == 'val':
            image_list = [l for l in image_list if is_train_fn(get_split_axis_loc(patch_locations[l]) - min_loc) and is_val_fn(get_split_axis_ortogonal_loc(patch_locations[l]) - min_orto_loc)]
        else:
            image_list = []

        if valid_img_names is not None and len(valid_img_names) > 0:
            image_list = [l for l in image_list if len([v for v in valid_img_names if v in l]) > 0]

        # parse gt list into dictionary of groundtruths
        self.gt_dict = self._load_gt_dict(root_dir)

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

        global_loc = self.get_image_global_location(self.image_list[index])

        sample = dict(image=image,
                      im_name=self.image_list[index],
                      im_size=im_size,
                      index=index, i=global_loc['i'], j=global_loc['j'])

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
        filename = os.path.basename(self.image_list[n])

        # load instances from loaded csv
        return self._load_gt_list(self.gt_dict[filename], self.fixed_bbox_size, self.resize_factor)

    @classmethod
    def _load_gt_dict(cls, gt_dir):
        gt_file = os.path.join(gt_dir, 'gt.pkl')

        with open(gt_file,'rb') as f:
            gt_dict = pickle.load(f)

        return gt_dict
    @classmethod
    def _load_gt_list(cls, gt_center_list, fixed_bbox_size=None, resize_factor=None):

        if len(gt_center_list) > 0:

            # add space for object size and switch X,Y position
            gt_center_list = np.concatenate((gt_center_list,np.zeros_like(gt_center_list)),axis=1)

            if resize_factor is not None:
                gt_center_list[:, :] *= resize_factor

            if fixed_bbox_size is not None:
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
        # {
        #     'name': 'RandomHorizontalFlip',
        #     'opts': {
        #         'keys': ('image', 'instance', 'label', 'ignore'), 'keys_bbox': ('center',),
        #         'p': 0,
        #     }
        # },
        # {
        #     'name': 'RandomVerticalFlip',
        #     'opts': {
        #         'keys': ('image', 'instance', 'label', 'ignore'), 'keys_bbox': ('center',),
        #         'p': 0,
        #     }
        # },
        # {
        #     'name': 'RandomResize',
        #     'opts': {
        #         'keys': ('image', 'instance', 'label', 'ignore'), 'keys_bbox': ('center',),
        #         'interpolation': (InterpolationMode.BILINEAR, InterpolationMode.NEAREST, InterpolationMode.NEAREST,
        #                           InterpolationMode.NEAREST),
        #         'scale_range': [0.8, 1.2]
        #     }
        # },
        # {
        #     'name': 'RandomCrop',
        #     'opts': {
        #         'keys': ('image', 'instance', 'label', 'ignore'), 'keys_bbox': ('center',),
        #         'size': (1 * 256, 1 * 256),
        #         'pad_if_needed': True
        #     }
        # },
        {
            'name': 'ToTensor',
            'opts': {
                'keys': ('image', 'instance', 'label', 'ignore'),
                'type': (torch.FloatTensor, torch.ShortTensor, torch.ByteTensor, torch.ByteTensor),
            }
        },
    ])

    plt.ion()
    total_test, total_train, total_val = [], [], []

    #db_params = dict(root_dir='/storage/datasets/tree_counting_dataset/croped_dataset_512x512',name='Acacia_06',
    #                 num_folds=3, fold_split_axis_rotate=90, fold_split_axis='i', PATCH_SIZE=512, resize_factor=1, fixed_bbox_size=20,
    #                 MAX_NUM_CENTERS=2048, BORDER_MARGIN_FOR_CENTER=0, transform=transform)
    #db_params = dict(root_dir='/storage/datasets/tree_counting_dataset/croped_dataset_512x512',name='Acacia_12',
    #                 num_folds=3, fold_split_axis_rotate=0, fold_split_axis='i', PATCH_SIZE=512, resize_factor=1, fixed_bbox_size=20,
    #                 MAX_NUM_CENTERS=2048, BORDER_MARGIN_FOR_CENTER=0, transform=transform)
    db_params = dict(root_dir='/storage/datasets/tree_counting_dataset/croped_dataset_512x512',name='Oilpalm',
                     num_folds=3, fold_split_axis_rotate=0, fold_split_axis='j', PATCH_SIZE=512, VALIDATION_REGION_SIZE=4*2048, resize_factor=1, fixed_bbox_size=20,
                     MAX_NUM_CENTERS=2048, BORDER_MARGIN_FOR_CENTER=0, transform=transform)

    for _fold in [0,1,2]:
        db_train = TreeCountingDataset(type='train',  fold=_fold, **db_params)
        db_test = TreeCountingDataset(type='test',  fold=_fold, **db_params)
        db_val = TreeCountingDataset(type='val', fold=_fold, **db_params)

        train_i = [TreeCountingDataset.get_image_global_location(l)['i'] for l in db_train.image_list]
        train_j = [TreeCountingDataset.get_image_global_location(l)['j'] for l in db_train.image_list]

        test_i = [TreeCountingDataset.get_image_global_location(l)['i'] for l in db_test.image_list]
        test_j = [TreeCountingDataset.get_image_global_location(l)['j'] for l in db_test.image_list]

        val_i = [TreeCountingDataset.get_image_global_location(l)['i'] for l in db_val.image_list]
        val_j = [TreeCountingDataset.get_image_global_location(l)['j'] for l in db_val.image_list]


        C = np.zeros((int(max(train_i + test_i + val_i)) + 5, int(max(train_j + test_j + val_j)) + 5), dtype=np.uint8)

        for _i, _j in zip(test_i, test_j):
            C[int(_i):int(_i) + 512, int(_j):int(_j) + 512] |= 1

        for _i, _j in zip(train_i, train_j):
            C[int(_i):int(_i) + 512, int(_j):int(_j) + 512] |= 2

        for _i, _j in zip(val_i, val_j):
            C[int(_i):int(_i) + 512, int(_j):int(_j) + 512] |= 4

        plt.figure(); plt.imshow(C)

        total_test.append((test_i,test_j))
        total_train.append((train_i,train_j))

        print('fold %d: train=%d, test=%d, val=%d' % (_fold, len(db_train), len(db_test), len(db_val)))

    max_test = [(int(max(test_i)),int(max(test_j))) for test_i, test_j in total_test]
    max_test = [max(t) for t in (zip(*max_test))]

    C = np.zeros((max_test[0] + 5, max_test[1] + 5), dtype=np.uint8)

    for fold,(test_i, test_j) in enumerate(total_test):
        for _i, _j in zip(test_i, test_j):
            C[int(_i):int(_i) + 512, int(_j):int(_j) + 512] |= (1 << fold)

    plt.figure(); plt.imshow(C)
    db = TreeCountingDataset(type='test', fold=0, **db_params)
    for item in tqdm(db):
        if False and np.array(item['ignore']).sum() > 0:
            center = item['center']
            gt_centers = center[(center[:, 0] > 0) | (center[:, 1] > 0), :]

            plt.clf()

            plt.subplot(1,2,1)
            plt.imshow(item['image'].permute([1,2,0]))
            plt.plot(gt_centers[:, 0], gt_centers[:, 1], 'r.')

            plt.subplot(1, 2, 2)
            #plt.imshow(item['ignore'][0])
            plt.imshow(item['instance'][0])
            plt.plot(gt_centers[:, 0], gt_centers[:, 1], 'r.')
            plt.show(block=False)

    print("end")
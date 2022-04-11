import sys
import numpy as np
from PIL import Image
from skimage.segmentation import relabel_sequential
import skimage.draw
import cv2

import torch
from torch.utils.data import Dataset

from datasets.LockableSeedRandomAccess import LockableSeedRandomAccess

class SyntheticDataset(Dataset, LockableSeedRandomAccess):

    def __init__(self, length, image_size, transform=None, normalize=False,  MAX_NUM_CENTERS=1024,
                 max_instance_overlap=0, allow_out_of_bounds=True, num_instances=[1.,1.],
                 instance_size_relative=[0.1,0.25], instance_aspect_ratio=[0.1,1.], instance_rotation=[-180,180],
                 siblings_probability=0.5, siblings_count=[1, 5], siblings_resize_probability=0.5, siblings_resize_range=[0.1, 0.5]):

        self.length = length
        self.image_size = image_size
        self.transform = transform
        self.normalize = normalize

        # generate a list of seeds so that we generate the same data for each index
        self.seed_list = np.random.randint(sys.maxsize, size=length)

        self.random_settings = dict(num_instances=num_instances,
                                    instance_size_relative=instance_size_relative,
                                    instance_aspect_ratio=instance_aspect_ratio,
                                    instance_rotation=instance_rotation,
                                    siblings_probability=siblings_probability,
                                    siblings_count=siblings_count,
                                    siblings_resize_probability=siblings_resize_probability,
                                    siblings_resize_range=siblings_resize_range
        )

        self.max_instance_overlap = max_instance_overlap
        self.allow_out_of_bounds = allow_out_of_bounds

        self.MAX_NUM_CENTERS = MAX_NUM_CENTERS

    def lock_samples_seed(self, index_list):
        # regenerate seed for indexes that are not locked
        for index in list(set(range(self.length))-set(index_list)):
            self.seed_list[index] = np.random.randint(sys.maxsize)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        instance_polygon_list = None
        while instance_polygon_list is None:
            seed = self.seed_list[index]

            # generate instances as list of polygons
            instance_polygon_list = self._generate_new_instnaces(seed, self.image_size)

            # if we were unable to generate sufficient instance then reset seed
            if instance_polygon_list is None:
                print('Resetting seed for index=%d' % index)
                self.seed_list[index] = np.random.randint(sys.maxsize)

        instance, label, center = self._draw_instances(instance_polygon_list)

        # use seed as unique identifier
        filename = '%d' % seed

        sample = {}
        sample['im_name'] = filename
        sample['image'] = instance
        sample['instance'] = instance
        sample['label'] = label
        sample['center'] = center

        # transform
        if(self.transform is not None):
            sample = self.transform(sample)

        if self.normalize:
            sample['image'] = (np.array(sample['image']) - 128)/ 128  # normalize to [-1,1]

        sample['index'] = index
        return sample


    def _draw_instances(self, instance_polygon_list):
        pic = np.zeros(shape=self.image_size, dtype=np.uint8)

        for i,p in enumerate(instance_polygon_list):
            p = np.reshape(p,[-1,2])
            xx, yy = skimage.draw.polygon(p[:, 0], p[:, 1], shape=self.image_size)
            pic[xx, yy] = i+1

        instance_map = np.zeros(self.image_size, dtype=np.uint8)

        if len(instance_polygon_list) > self.MAX_NUM_CENTERS:
            raise Exception("Too small MAX_NUM_CENTERS (increase it manually!!)")

        center_list = np.zeros((self.MAX_NUM_CENTERS, 2), dtype=np.float)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(self.image_size, dtype=np.uint8)
        mask = pic > 0
        if mask.sum() > 0:
            ids, forward_map, _ = relabel_sequential(pic[mask])
            instance_map[mask] = ids
            class_map[mask] = 1

            # remap all centers as well using forward_map
            for n, polygon in enumerate(instance_polygon_list):
                id = forward_map[n + 1]
                if id > 0:
                    center_list[id] = polygon.reshape(-1,2).mean(axis=0)[[1,0]]

        return Image.fromarray(instance_map), Image.fromarray(class_map), center_list

    def _generate_new_instnaces(self, seed, img_size):
        rng = np.random.default_rng(seed)

        num_instances = rng.integers(self.random_settings['num_instances'][0],
                                     self.random_settings['num_instances'][1])

        instance_rect_list = []
        instance_polygon_list = []

        retry = 0

        primary_ids = []

        # repeat until enough instances
        while len(instance_polygon_list) < num_instances and retry < 1000:
            retry += 1
            if len(instance_polygon_list) <= 0 or rng.uniform() > self.random_settings['siblings_probability']:
                # generate new random instance
                instance_rect, polygon = self._generate_single_instance(rng, img_size)

                # check that instance is valid (does not overlap other instances more than allowed etc)
                if not self._is_valid_instance(instance_rect, polygon,
                                               instance_rect_list, instance_polygon_list,
                                               img_size):
                    continue

                # add to list
                instance_polygon_list.append(polygon.reshape(-1))
                instance_rect_list.append(instance_rect)
                # save as primary id so that siblings are created only from them
                primary_ids.append(len(instance_rect_list)-1)
            else:
                # generate sibling instances based on existing one
                siblings_rect_list, siblings_polygon_list = self._generate_sibling_instance(rng, primary_ids,
                                                                                            instance_rect_list,
                                                                                            instance_polygon_list,
                                                                                            img_size)

                # retry if cannot create enough siblings
                if siblings_rect_list is None or siblings_polygon_list is None:
                    continue

                instance_polygon_list.extend(siblings_polygon_list)
                instance_rect_list.extend(siblings_rect_list)

            retry = 0

        # if we have enough all instances (primary+siblings) then continue
        if len(instance_polygon_list) < num_instances:
            return None

        return np.array(instance_polygon_list)

    def _generate_sibling_instance(self, rng, all_primary_ids, instance_rect_list, instance_polygon_list, img_size):
        d = np.sqrt(img_size[0] ** 2 + img_size[1] ** 2)

        primary_id = rng.choice(all_primary_ids)

        (primary_center, (primary_w, primary_h), primary_rotation) = instance_rect_list[primary_id]

        # chose how many siblings to generate
        num_siblings = rng.integers(self.random_settings['siblings_count'][0],
                                    self.random_settings['siblings_count'][1])

        allowed_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        allowed_directions_length = [primary_w / 2, primary_h / 2, primary_w / 2, primary_h / 2]

        siblings_polygon_list = []
        siblings_rect_list = []

        retry_siblings = 0
        while len(siblings_polygon_list) < num_siblings and retry_siblings < 1000:
            retry_siblings += 1

            # rotation of each sibling remains the same
            rotation = primary_rotation
            rotation_rad = rotation * np.pi / 180

            # by default use the same size
            w, h = primary_w, primary_h

            # but resize it if required
            if rng.uniform() < self.random_settings['siblings_resize_probability']:
                resize_factor = rng.uniform(self.random_settings['siblings_resize_range'][0],
                                            self.random_settings['siblings_resize_range'][1])

                w, h = resize_factor * w, resize_factor * h

            # make sure instance is not smaller than allowed
            if np.sqrt(w**2 + h**2) < self.random_settings['instance_size_relative'][0] * d:
                continue

            # chose direction
            direction_id = rng.integers(4)

            # prepare new direction length based on size of new instance
            add_direction_length = w if direction_id % 2 == 0 else h
            # also add 1px margin
            add_direction_length += 1

            # new center is defined as old center + (direction times (w/2 + new_primary_w/2) times rotation)
            direction_scaled = np.array(allowed_directions[direction_id]) * \
                               (allowed_directions_length[direction_id] + add_direction_length / 2)

            direction_scaled_rotated = np.matmul(np.array([[np.cos(rotation_rad), -np.sin(rotation_rad)],
                                                           [np.sin(rotation_rad), np.cos(rotation_rad)]]),
                                                 direction_scaled)

            cx, cy = np.array(primary_center) + direction_scaled_rotated

            instance_rect = ((int(cx), int(cy)), (int(w), int(h)), rotation)
            polygon = cv2.boxPoints(instance_rect)

            # check that instance is valid (does not overlap other instances more than allowed etc)
            if not self._is_valid_instance(instance_rect, polygon,
                                           instance_rect_list+siblings_rect_list,
                                           instance_polygon_list+siblings_polygon_list,
                                           img_size):
                continue

            # prepare direction for other instances (to indicated that at this position there is already instance)
            allowed_directions_length[direction_id] += add_direction_length

            # add to list
            siblings_polygon_list.append(polygon.reshape(-1))
            siblings_rect_list.append(instance_rect)

            retry_siblings = 0

        # if cannot return all siblings then return None
        if len(siblings_rect_list) < num_siblings:
            return None, None

        return siblings_rect_list, siblings_polygon_list

    def _generate_single_instance(self, rng, img_size):
        d = np.sqrt(img_size[0] ** 2 + img_size[1] ** 2)

        cx, cy = rng.integers(img_size[0]), rng.integers(img_size[1])
        instance_size = rng.uniform(self.random_settings['instance_size_relative'][0],
                                    self.random_settings['instance_size_relative'][1]) * d

        aspect_ratio = rng.uniform(self.random_settings['instance_aspect_ratio'][0],
                                   self.random_settings['instance_aspect_ratio'][1])

        w = instance_size * aspect_ratio * np.sqrt(1 / (aspect_ratio ** 2 + 1))
        h = instance_size * np.sqrt(1 / (aspect_ratio ** 2 + 1))

        rotation = rng.uniform(self.random_settings['instance_rotation'][0],
                               self.random_settings['instance_rotation'][1])

        # store instance as rotated rectange in opencv2 format
        instance_rect = ((int(cx), int(cy)), (int(w), int(h)), rotation)
        # convert generated instance to polygon
        polygon = cv2.boxPoints(instance_rect)

        return instance_rect, polygon

    def _is_valid_instance(self, instance_rect, polygon, existing_instances, existing_polygons, img_size):
        # find overlap ration with any existing instances
        overlap_ratio = 0
        for r, p in zip(existing_instances, existing_polygons):
            r1 = cv2.rotatedRectangleIntersection(instance_rect, r)
            if r1[0] > 0:
                area = cv2.contourArea(r1[1])
                area_ratio = area / (cv2.contourArea(polygon) + cv2.contourArea(p.reshape(-1, 2)) - area)
                overlap_ratio = max(overlap_ratio, area_ratio)

        # make sure there is no or minimal overlap with other instance
        if overlap_ratio > self.max_instance_overlap:
            return False

        # make sure all points are within image
        if not self.allow_out_of_bounds:
            if (polygon < 0).any() or (polygon[:, 0] >= img_size[0]).any() or (polygon[:, 1] >= img_size[1]).any():
                return False

        return True
if __name__ == "__main__":
    db = SyntheticDataset(length=100,
                          image_size=(512,512),
                          max_instance_overlap=0,
                          allow_out_of_bounds=True,
                          num_instances=[1.,10.],
                          instance_size_relative=[0.05,0.5],
                          instance_aspect_ratio=[0.1,0.3],
                          instance_rotation=[-180,180])
    for item in db:
        pass
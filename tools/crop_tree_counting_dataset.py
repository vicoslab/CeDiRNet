import os
from PIL import Image

from tqdm import tqdm

import numpy as np
import pickle

# enable loading of large files
Image.MAX_IMAGE_PIXELS = None

class PatchInclusionTest:
    def __init__(self, output_image_size, roi=None, roi_resize_factor=(10,10)):
        self.output_image_size = output_image_size

        self.roi = roi
        self.roi_resize_factor = roi_resize_factor
        if self.roi_resize_factor is not None:
            self.w = int(self.output_image_size[1] / self.roi_resize_factor[1])
            self.h = int(self.output_image_size[0] / self.roi_resize_factor[0])

    def __call__(self, patch_xy, gt_centers, B=5, only_positive_patches=True, skip_boundry_objects=False):
        if self.roi is None:
            patch_inclusion = _calc_patch_inclusion(patch_xy, self.output_image_size, gt_centers, B)

            if only_positive_patches and any(patch_inclusion == 1) is False:
                return False

            if skip_boundry_objects and any((patch_inclusion > 0) * (patch_inclusion < 4)):
                return False

            return True
        else:
            x,y = int(patch_xy[1] / self.roi_resize_factor[1]), int(patch_xy[0] / self.roi_resize_factor[0])

            return (self.roi[x:x+self.w,y:y+self.h] == 0).any()

def _calc_patch_inclusion(patch_xy, patch_size, gt_centers, B=5):
    patch_inclusion = (gt_centers[:, 0] >= patch_xy[0] - B) * \
                      (gt_centers[:, 0] <= patch_xy[0] + patch_size[0] + B) * \
                      (gt_centers[:, 1] >= patch_xy[1] - B) * \
                      (gt_centers[:, 1] <= patch_xy[1] + patch_size[1] + B)

    return patch_inclusion



if __name__ == "__main__":
    DATASET_DIR = '/path/to/datasets/'
    OUTPUT_DIR = '/path/to/datasets/cropped'

    FILES = [(os.path.join(DATASET_DIR,'Acacia/12Months_Crop.jpg'),os.path.join(DATASET_DIR,'Acacia/12Months.csv')),
             (os.path.join(DATASET_DIR,'Acacia/6Months_Crop.jpg'),os.path.join(DATASET_DIR,'Acacia/06Months.csv')),
             (os.path.join(DATASET_DIR,'Oilpalm/oilpalm-insight-2.jpg'),os.path.join(DATASET_DIR,'Oilpalm/oilpalm_center.txt'))]

    resize_factor = 1.0
    output_image_size = (512,512)
    steps = (128*3,128*3)

    os.makedirs(OUTPUT_DIR,exist_ok=True)

    for f,gt in FILES:
        all_gt_centers = {}

        filename = os.path.basename(f)
        folder = os.path.basename(os.path.dirname(f))
        print('Processing image %s' % filename)

        try:
            gt_centers = np.loadtxt(gt, delimiter=',', skiprows=1)
        except:
            gt_centers = np.loadtxt(gt, delimiter=' ')
        gt_centers = gt_centers[:,-2:]

        I = Image.open(f)
        imsize = I.size

        roi_filename = f.replace(".jpg","-roi.jpg")
        if os.path.exists(roi_filename):
            roi = Image.open(roi_filename)

            roi_resize_factor = imsize[1]/roi.size[1], imsize[0]/roi.size[0]

            test_patch = PatchInclusionTest(output_image_size, np.array(roi), roi_resize_factor)
        else:
            test_patch = PatchInclusionTest(output_image_size)

        new_patches_list = [(i, j) for j in np.arange(0, imsize[1] - output_image_size[1] + steps[1], steps[1])
                                for i in np.arange(0, imsize[0] - output_image_size[0] + steps[0], steps[0])
                                    if test_patch((i, j), gt_centers, B=5, only_positive_patches=True)]


        for i,j in tqdm(new_patches_list):
            x,y,w,h = i,j,output_image_size[0], output_image_size[1]

            patch_name = '%s_%s_patch%dx%d_i=%d_j=%d.jpg' % (folder, os.path.splitext(filename)[0], w, h, x, y)
            patch_img = I.crop((i,j, i+w, j+h))

            patch_inclusion = _calc_patch_inclusion((x,y), output_image_size, gt_centers, B=0)

            all_gt_centers[patch_name] = gt_centers[patch_inclusion,:]
            all_gt_centers[patch_name][:, 0] -= x
            all_gt_centers[patch_name][:, 1] -= y
            if resize_factor != 1.0:
                newsize = int(w*resize_factor),int(h*resize_factor)
                patch_img = patch_img.resize(newsize)
                all_gt_centers[patch_name][:, 0] *= newsize[0] / w
                all_gt_centers[patch_name][:, 1] *= newsize[1] / h

            patch_img.save(os.path.join(OUTPUT_DIR,patch_name), 'JPEG')


        with open(os.path.join(OUTPUT_DIR, 'gt.pkl'),'wb') as f:
            pickle.dump(all_gt_centers, f)
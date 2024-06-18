import os
import numpy as np
import albumentations as al
import rasterio
import cv2
from rasterio.windows import Window
import kornia
from torch.utils.data import Dataset
import torch

class CarSegmentation(Dataset):
    def __init__(self, image_dir, seg_dir, transform=None):
        self.width = 7952
        self.height = 5304
        self.out_shape = (400, 400)
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.filenames = [os.path.join(seg_dir, filename) for filename in os.listdir(seg_dir)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = os.path.basename(self.filenames[index])

        with rasterio.open(os.path.join(self.image_dir, filename)) as src_image:
            with rasterio.open(self.filenames[index]) as src_mask:
                while True:
                    tmp_i, tmp_m = self.random_crop(src_image, src_mask)
                    if tmp_m.max() == 1:
                        image = tmp_i
                        mask = tmp_m
                        break

        mask = np.stack([mask, 1 - mask], axis=-1).astype(np.float32)
        sample = {'image': image, 'masks': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def random_crop(self, image, mask):

        i = np.random.randint(0, self.height - self.out_shape[0])
        j = np.random.randint(0, self.width - self.out_shape[1])

        image_window = image.read(window=Window(j, i, self.out_shape[1], self.out_shape[0]))
        mask_window = mask.read(window=Window(j, i, self.out_shape[1], self.out_shape[0]))[0] / 255

        return image_window.transpose(1, 2, 0), mask_window

class ToTensor(object):
    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        return {'image': kornia.image_to_tensor(image.copy()),
                'masks': kornia.image_to_tensor(masks.copy())}

class Augmenter:
    def __init__(self):
        self.transform = al.Compose([
            al.HorizontalFlip(p=0.5),
            al.VerticalFlip(p=0.5),
            al.Rotate(p=1, rotate_method="ellipse",border_mode=cv2.BORDER_CONSTANT),
            al.Transpose(p=0.5),
            al.RandomBrightnessContrast(brightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1)),
            #al.CenterCrop(320,320,1)
        ], p=1)

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        transformed = self.transform(image=image, mask=masks)
        print(transformed['image'].shape)
        return {'image': transformed['image'],
                 'masks':transformed['mask']}

class Normalizer(object):
    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        return {'image': (image.astype(np.float32) / 255 * 2) - 1, 'masks': masks}

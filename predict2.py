import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as al
import cv2

from osgeo import gdal
from time import perf_counter

from networks.modeling import deeplabv3plus_mobilenet

device = torch.device("cpu")

model = deeplabv3plus_mobilenet(num_classes=2)

checkpoint = r"E:\browser_downloads\train_titanium_extended3_600.pth"
# checkpoint = r"E:\AI\save_dir2\checkpoints\lithiumv8\train_lithiumV8_2_1200.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))

model = model.to(device)
model.eval()

filename = r"E:\AI\dataset6\2021_07_01_RX2_g201b20406_f031_0658g.tif"
filename_gt = r"E:\AI\dataset6g\2021_07_01_RX2_g201b20406_f031_0658g.tif"

x = 3500
y = 2800
crop = 1500

image_gdal = gdal.Open(filename)
image_gdal_gt = gdal.Open(filename_gt)
gt = image_gdal_gt.ReadAsArray(x, y, crop, crop).astype(np.float32)
image = image_gdal.ReadAsArray(x, y, crop, crop).astype(np.float32)
#image = image.transpose((0, 2, 1))

t0 = perf_counter()
image = (image.astype(np.float32) / 255 * 2) - 1
head = model(torch.from_numpy(image[np.newaxis]))
head = F.sigmoid(head)
print(perf_counter() - t0)

head = head.detach().cpu().numpy()[0, 0]

def calculate_iou(pred, target, threshold=0.3):
    pred = (pred > threshold).astype(np.float32)
    target = (target > threshold).astype(np.float32)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    fp = (np.logical_and(pred == 1, target == 0).sum()) / union
    fn = (np.logical_and(pred == 0, target == 1).sum()) / union
    iou = intersection / union
    if union == 0:
        return float('nan')  # If there is no union, IoU is undefined
    return iou, fp, fn


miou, fp, fn = calculate_iou(head, gt[-1])
print(f"mIoU: {miou:.4f}")
print(f"False Positives: {fp:.4f}")
print(f"False Negatives: {fn:.4f}")
plt.matshow(head)
plt.matshow(gt[-1])
plt.matshow(image[0])
plt.show()
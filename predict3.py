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

device = torch.device("cpu")

model = deeplabv3plus_mobilenet(num_classes=2)

checkpoint = r"E:\AI\save_dir2\checkpoints\phosphorusV9\train_phosphorusV9_2_1200.pth"

# checkpoint = r"E:\AI\save_dir2\checkpoints\lithiumv8\train_lithiumV8_2_1200.pth"
model.load_state_dict(torch.load(checkpoint, map_location=device))

model = model.to(device)
model.eval()

filename = r"E:\AI\dataset6\2021_07_01_RX2_g201b20406_f031_0658g.tif"
filename_gt = r"E:\AI\dataset6g\2021_07_01_RX2_g201b20406_f031_0658g.tif"
crops =[(r"E:\AI\dataset6\2021_07_01_RX2_g201b20406_f031_0658g.tif",
         r"E:\AI\dataset6g\2021_07_01_RX2_g201b20406_f031_0658g.tif",
         3500,
         2800),
        (r"E:\AI\dataset6\2021_07_01_RX2_g201b20406_f031_0314.tif",
         r"E:\AI\dataset6g\2021_07_01_RX2_g201b20406_f031_0314.tif",
         3500,
         2800),
        (r"E:\AI\dataset6\2021_06_28_RX2_g201b20406_f027_0488.tif",
         r"E:\AI\dataset6g\2021_06_28_RX2_g201b20406_f027_0488.tif",
         3500,
         2800),
        ]
x = 3500
y = 2800
crop = 1500
global_miou=0
global_fp=0
global_fn=0
for obj in crops:
    image_gdal = gdal.Open(obj[0])
    image_gdal_gt = gdal.Open(obj[1])
    gt = image_gdal_gt.ReadAsArray(obj[2], obj[3], crop, crop).astype(np.float32)
    image = image_gdal.ReadAsArray(obj[2], obj[3], crop, crop).astype(np.float32)
    image = (image.astype(np.float32) / 255 * 2) - 1
    head = model(torch.from_numpy(image[np.newaxis]))
    head = F.sigmoid(head)
    head = head.detach().cpu().numpy()[0, 0]
    miou, fp, fn = calculate_iou(head, gt[-1])
    global_miou += miou
    global_fp += fp
    global_fn += fn
    print(f"mIoU: {miou:.4f}")
    print(f"False Positives: {fp:.4f}")
    print(f"False Negatives: {fn:.4f}")
    plt.matshow(head)
    plt.matshow(gt[-1])
    plt.matshow(image[0])
    plt.show()
global_miou /= len(crops)
global_fp /= len(crops)
global_fn /= len(crops)
print("\n",f"mIoU: {global_miou:.4f}", f"  False Positives: {global_fp:.4f}", f"  False Negatives: {global_fn:.4f}")








import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import albumentations as al
import cv2
from PIL import Image

from loguru import logger

import pathlib

from osgeo import gdal
from time import perf_counter

from TifMaker import Tiff
from networks.modeling import deeplabv3plus_mobilenet

def CarsAi():
    device = torch.device("cpu")

    model = deeplabv3plus_mobilenet(num_classes=2)

    checkpoint = r"E:\AI\save_dir2\checkpoints\phosphorusV11\train_phosphorusV11_2_1200.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    model = model.to(device)
    model.eval()
    image = Image.open("image.jpg")
    p = pathlib.Path("E:/","AI","segmetation2","segmetation3")
    image.save(fp=str(p) + "\\image.tif",format="TIFF")
    filename = r'image.tif'
    filename_gt = r"E:\AI\dataset6g\2021_07_01_RX2_g201b20406_f031_0658g.tif"

    x = 0
    y = 0
    crop = 3000

    image_gdal = gdal.Open(filename)
    image = image_gdal.ReadAsArray(x, y, crop, crop).astype(np.float32)

    t0 = perf_counter()
    image = (image.astype(np.float32) / 255 * 2) - 1
    head = model(torch.from_numpy(image[np.newaxis]))
    head = F.sigmoid(head)

    logger.info(f"Time: {perf_counter() - t0}")

    head = head.detach().cpu().numpy()[0, 0]

    head_rescaled = (head * 255).astype(np.uint8)

    head_image = Image.fromarray(head_rescaled)
    head_image.save('head_output.jpg')

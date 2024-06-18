import os

import torch
import kornia
import xml2dict
import numpy as np
import albumentations as al

from PIL import Image, ImageDraw
from glob import glob
from osgeo import gdal
from torch.utils.data import Dataset
from dataset import CarSegmentation

save_dir1 = r"E:\AI\dataset_last2g"
with open(r"E:\browser_downloads\annotations.xml", 'r') as file:
    data = xml2dict.parse(file.read())

i = 0
images = data['annotations']['image'] 
if isinstance(images, dict):
    images = [images]

for annotation in images:
    mask = Image.new('RGB', (int(annotation['@width']), int(annotation['@height'])), (0, 0, 0))
    if annotation.get('polygon'):
        name = os.path.basename(annotation['@name'])
        pdraw = ImageDraw.Draw(mask)
        polygons = annotation['polygon']
        if isinstance(polygons, dict):
            polygons = [polygons]
        for polyline in polygons:
            points = list(map(lambda x: tuple(map(lambda y: round(float(y)), x.split(','))), polyline['@points'].split(';')))
            pdraw.polygon(points, fill=(255, 255, 255), outline=(255, 255, 255))
        i += 1
        print(i)
        mask.save(os.path.join(save_dir1, name))
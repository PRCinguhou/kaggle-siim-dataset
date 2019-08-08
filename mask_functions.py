import numpy as np
import pydicom
from os.path import join
import os
from pydicom.data import get_testdata_files
import matplotlib.pyplot as plt
import pandas as pd

def read_dicom(img_path):

    ds = pydicom.dcmread(img_path)
    data = ds.pixel_array
    h, w = data.shape
    data = data.reshape(h,w,1)
    data = np.repeat(data, 3, axis=2)
    return data, h, w

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height).astype(np.float32)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    if starts[0] == -1:
        return mask.reshape(width, height)
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)

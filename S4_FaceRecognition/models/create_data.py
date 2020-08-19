from torch import nn, optim, as_tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.init import *
from torchvision import transforms, utils, datasets, models
from models.inception_resnet_v1 import InceptionResnetV1
import cv2
from PIL import Image
from pdb import set_trace
import time
import copy
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import io, transform
from tqdm import trange, tqdm
import csv
import glob
import dlib
import pandas as pd
import numpy as np

from IPython.display import Video
Video("data/IMG_2411.MOV", width=200, height=350)

vidcap = cv2.VideoCapture('IMG_2411.MOV')
success,image = vidcap.read()
count = 0
success = True
while success:
    cv2.imwrite(f"./Michael_Chaykowsky/Michael_Chaykowsky_{
                  format(count, '04d') }.png", image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

%%!
for szFile in ./Michael_Chaykowsky/*.png
do 
    magick mogrify -rotate 90 ./Michael_Chaykowsky/"$(basename "$szFile")" ; 
done